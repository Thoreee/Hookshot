"""The write-back diff keeps what the CPU wrote during a part where the GPU left a word alone (gpu/vk/vk_memory.cpp, "the diff's
shadow as it was before the part").

The game's threads run a frame ahead of the GPU: a particle emitter writes a new particle's record into a free slot of the buffer
the effect shaders simulate while the GPU still runs the frame before. The diff read a page the CPU wrote from guest memory into
its shadow at the end of the part, new particle and all, found the mirror's old slot different from it and wrote that back over
the particle: Kokiri Forest's pollen mostly went (all there with RX_VK_MIRROR=0). Now the pages a part writes back are pinned at
its end, and a pinned page the shadow does not know is copied from the mirror in the part's uploads, before its GPU work.

This takes shadow_pin and its helpers (wb_snapshot_on to shadows_pin_writebacks, shadows_unpin_replay, shadow_pins_clear,
shadows_part_begin, shadows_before_segments) and the shadows' report functions from the source as they are, and runs a part
through them with a pool, a mirror, a shadow and guest memory of the test's own. The part's end is the test's model of
memory_end_submission (the order of its calls is checked against the source below): the snapshot copies run before the GPU's
work, the unpinned pages the shadow does not know are read from guest memory, then the words that differ are written
(shaders/wb_diff.comp). It runs once as is and once with RX_VK_WB_SNAPSHOT=0, where the old loss must show.

python -I -B tests/test_wb_diff_cpu_spawns.py [--cxx g++]
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

RUN_ALL_ARGS = ["--cxx", "{cxx}"]

ROOT = Path(__file__).resolve().parents[1]


def block(text, marker):
    start = text.index(marker)
    end = text.index('{', start) + 1
    depth = 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def check_order(text):
    """The calls the model below stands for, where the source makes them"""
    end = block(text, 'void memory_end_submission(VkCommandBuffer cmd, uint64_t part) {')
    pin, snap = end.index('if (diff) shadows_pin_writebacks(), '), end.index('shadows_snap_record(), ')
    unpin, forget = end.index('shadows_unpin_replay(), '), end.index('shadows_forget_written(), ')
    fill, clear = end.index('shadow_is_pinned(*p, pg)) continue;'), end.index('shadow_pins_clear();')
    assert pin < snap < unpin < forget < fill < clear, 'memory_end_submission: pin, snapshots, unpin, forget, fill, clear, in order'
    assert end.index('std::sort(writebacks.begin()') < pin, 'the ranges pinned once sorted (merged runs)'
    assert end.rindex('writebacks.clear();') < clear, 'the pins let go after the write-back'
    assert 'shadows_part_begin();' in block(text, 'void memory_begin_submission() {')
    pre = block(text, 'void memory_sync_mirrors_before_segments() {')
    assert pre.index('shadows_before_segments();') < pre.index('if (off) return;'), 'before segments, RX_VK_NO_PRESYNC or not'
    dev = (ROOT / 'docs/development.md').read_text(encoding='utf-8')
    assert '`RX_VK_WB_SNAPSHOT=0`' in dev, 'docs/development.md: the switch documented with the other write-back ones'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--source', type=Path, default=ROOT / 'gpu/vk/vk_memory.cpp')
    args = parser.parse_args()
    text = args.source.read_text(encoding='utf-8')
    check_order(text)
    pins = text[text.index('bool wb_snapshot_on() {'):text.index('// (pools_mu held, at the part\'s end) the pinned pages\' copies')]
    helper = chr(10).join(block(text, marker) for marker in (
        'void shadow_forget(Pool& p, uint64_t at, uint64_t len) {',
        'void shadows_forget_written() {',
        'void shadow_written_to(std::vector<uint32_t>& list, const uint32_t* pages, int64_t n) {',
        'void shadow_written(const uint32_t* pages, int64_t n) {',
        'int guest_cpu_write(uint64_t gpu, const void* src, uint64_t n) {',
        'void guest_cpu_wrote(uint64_t gpu, uint64_t n) {')) + chr(10) + pins + chr(10).join(block(text, marker) for marker in (
        'void shadows_unpin_replay() {',
        'void shadow_pins_clear() {',
        'void shadows_part_begin() {',
        'void shadows_before_segments() {'))
    fixture = r'''
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>
#define CHECK(ok) do { if (!(ok)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #ok); std::exit(1); } } while (0)
constexpr uint64_t kPageShift = 12;
// (the test's pool: 8 pages of guest memory at kLo, a mirror and a shadow of it, in words)
constexpr uint64_t kLo = 0x10000000, kPages = 8, kWords = kPages * 1024;
typedef struct VkBuffer_T* VkBuffer;
VkBuffer const kShadow = (VkBuffer)0x5, kMirror = (VkBuffer)0x6;
struct Pool {
    uint64_t lo = kLo, hi = kLo + kPages * 4096;
    VkBuffer shadow = kShadow, mirror = kMirror;
    std::vector<uint64_t> shadow_known = std::vector<uint64_t>(1, 0);
    std::vector<uint64_t> shadow_pinned;
};
std::map<uint64_t, Pool> imported;
struct WriteBack {
    Pool* pool;
    uint64_t lo, hi;   // offsets in the pool
};
std::vector<WriteBack> writebacks;   // (the part's written ranges, as gpu_buffer_bind lists them)
uint64_t shadow_bytes = kPages * 4096;
std::atomic<bool> par_active{false};
uint64_t mirror_epoch = 0;
std::vector<uint32_t> guest(kWords), mirror(kWords), shadow(kWords);
struct Host {   // (GPU addresses are guest addresses here)
    static uint64_t gpu_guest(uint64_t gpu) { return gpu >= kLo && gpu < kLo + kPages * 4096 ? gpu : 0; }
    static int gpu_write(uint64_t gpu, const void* src, uint64_t n) {
        if (!gpu_guest(gpu) || !gpu_guest(gpu + n - 1)) return 0;
        std::memcpy((uint8_t*)guest.data() + (gpu - kLo), src, n);
        return 1;
    }
};
Host the_host;
Host* host = &the_host;
bool wb_diff_on() { return true; }
std::mutex shadow_written_mu;
std::vector<uint32_t> shadow_written_pages, shadow_written_collected_list, shadow_written_by_replay;
bool shadow_written_all = false;
int64_t st_wbd_known_game = 0, st_wbd_known_collected = 0, st_wbd_known_replay = 0;
void shadow_written_to(std::vector<uint32_t>& list, const uint32_t* pages, int64_t n);
void guest_cpu_wrote(uint64_t gpu, uint64_t n);
void shadows_snap_record();
''' + helper + r'''
Pool& pool() { return imported.begin()->second; }
// the copies shadows_snap_record records run in the part's uploads, before its GPU work: from the mirror as the work found it
std::vector<uint32_t> mirror_start(kWords);
void shadows_snap_record() {
    for (const ShadowRun& r : shadow_snaps)
        if (r.p->shadow == r.shadow)
            std::copy(mirror_start.begin() + r.at / 4, mirror_start.begin() + (r.at + r.len) / 4, shadow.begin() + r.at / 4);
}
// the CPU's writes the next part's mirror sync copies (whole pages: as a capture holds them)
std::vector<bool> dirty(kPages);
void game_write(uint64_t w, uint32_t v) {   // (the game's thread: in a capture, reported)
    guest[w] = v, dirty[w / 1024] = true;
    const uint32_t pg = (uint32_t)((kLo >> 12) + w / 1024);
    shadow_written(&pg, 1);
}
// a part: begun, its mirror synced, its range [at, at + len) bound written; the GPU's work then starts
uint64_t part_at = 0, part_len = 0;
void part_begin(uint64_t at, uint64_t len) {
    mirror_epoch++;
    shadows_part_begin();
    for (uint64_t pg = 0; pg < kPages; pg++)
        if (dirty[pg]) std::copy(guest.begin() + pg * 1024, guest.begin() + (pg + 1) * 1024, mirror.begin() + pg * 1024), dirty[pg] = false;
    writebacks.push_back({&pool(), at, at + len});
    part_at = at, part_len = len;
}
void gpu_starts() { mirror_start = mirror; }
int64_t filled = 0;
void part_end() {   // (memory_end_submission's write-back of the part's range, with the diff)
    Pool& p = pool();
    shadows_pin_writebacks();
    shadows_snap_record();
    shadows_unpin_replay();
    shadows_forget_written();
    for (uint64_t pg = part_at >> kPageShift; pg <= (part_at + part_len - 1) >> kPageShift; pg++) {
        uint64_t& bits = p.shadow_known[pg >> 6];
        if (bits & (1ull << (pg & 63)) || shadow_is_pinned(p, pg)) continue;
        bits |= 1ull << (pg & 63);
        std::copy(guest.begin() + pg * 1024, guest.begin() + (pg + 1) * 1024, shadow.begin() + pg * 1024);
        filled++;
    }
    for (uint64_t w = part_at / 4; w < (part_at + part_len) / 4; w++)
        if (mirror[w] != shadow[w]) guest[w] = mirror[w], shadow[w] = mirror[w];
    writebacks.clear();
    shadow_pins_clear();
}
void reset() {
    imported.clear();
    imported.emplace(kLo, Pool{});
    std::fill(guest.begin(), guest.end(), 0u), std::fill(mirror.begin(), mirror.end(), 0u), std::fill(shadow.begin(), shadow.end(), 0u);
    std::fill(dirty.begin(), dirty.end(), false);
    shadow_written_pages.clear(), shadow_written_collected_list.clear(), shadow_written_by_replay.clear();
    shadow_written_all = false;
    shadow_pins.clear(), shadow_snaps.clear(), writebacks.clear();
    pool().shadow_known[0] = (1ull << kPages) - 1;   // (every page known: the shadow, the mirror and guest memory at 0)
    filled = 0;
}
bool pins_gone() {
    for (uint64_t b : pool().shadow_pinned) if (b) return false;
    return shadow_pins.empty() && shadow_snaps.empty();
}
int main(int argc, char** argv) {
    const bool snapshot = argc > 1 && argv[1][0] == '1';   // (RX_VK_WB_SNAPSHOT as the runner set it)
    CHECK(wb_snapshot_on() == snapshot);
    if (snapshot) {   // a pool past 64 pages: pinned 64 pages at a time, a range ending inside a page, a last page not whole
        mirror_epoch = 1;
        Pool big;
        big.hi = big.lo + 300 * 4096 - 100;
        big.shadow_known.assign(5, 0);
        big.shadow_known[1] = 1ull << 6;                 // (page 70 known: pinned, not copied)
        shadow_pin(big, 70 * 4096 + 100, 120 * 4096);   // (pages 70 to 190)
        for (uint64_t pg = 0; pg < 300; pg++) CHECK(shadow_is_pinned(big, pg) == (pg >= 70 && pg <= 190));
        CHECK(shadow_pins.size() == 1 && shadow_pins[0].at == 70 * 4096 && shadow_pins[0].len == 121 * 4096);
        CHECK(shadow_snaps.size() == 1 && shadow_snaps[0].at == 71 * 4096 && shadow_snaps[0].len == 120 * 4096);
        shadow_pin(big, 0, 300 * 4096);   // (all of it again: the rest pinned, the range kept whole, as long as the pool)
        CHECK(shadow_pins.size() == 2 && shadow_pins[1].at == 0 && shadow_pins[1].len == 300 * 4096 - 100);
        CHECK(shadow_snaps.size() == 3 && shadow_snaps[1].at == 0 && shadow_snaps[1].len == 70 * 4096);   // (copied: the new ones)
        CHECK(shadow_snaps[2].at == 191 * 4096 && shadow_snaps[2].at + shadow_snaps[2].len == 300 * 4096 - 100);
        for (uint64_t pg = 0; pg < 300; pg++) CHECK(shadow_is_pinned(big, pg) && (big.shadow_known[pg >> 6] >> (pg & 63) & 1));
        const size_t pins = shadow_pins.size();
        shadow_pin(big, 300 * 4096, 4096);   // (past the pool: nothing)
        shadow_pin(big, 5 * 4096, 200 * 4096);   // (all pinned already: nothing)
        CHECK(shadow_pins.size() == pins);
        shadow_pins_clear();
        for (uint64_t pg = 0; pg < 300; pg++) CHECK(!shadow_is_pinned(big, pg));
        // the part's list: a range for each binding, the same ones again and overlapping ones, merged before they are pinned
        writebacks = {{&big, 4096, 3 * 4096}, {&big, 4096, 3 * 4096}, {&big, 2 * 4096 + 8, 5 * 4096}, {&big, 10 * 4096, 11 * 4096}};
        shadows_pin_writebacks();
        CHECK(shadow_pins.size() == 2 && shadow_pins[0].at == 4096 && shadow_pins[0].len == 4 * 4096);
        for (uint64_t pg = 0; pg < 300; pg++) CHECK(shadow_is_pinned(big, pg) == ((pg >= 1 && pg <= 4) || pg == 10));
        writebacks.clear();
        shadow_pins_clear();
    }
    // the pollen: a slot of a page the shadow knows; the GPU moves particle A, the game spawns B into a free slot meanwhile and
    // its next capture reports the page before the part's end
    for (int known = 1; known >= 0; known--) {
        reset();
        if (!known) game_write(3 * 1024 + 900, 4);   // (or one the CPU wrote before the part: synced, and unknown to the shadow)
        part_begin(3 * 4096, 4096);
        gpu_starts();
        mirror[3 * 1024 + 10] = 5;        // (the GPU's work: particle A moved)
        game_write(3 * 1024 + 20, 7);     // (the game's thread: particle B, into a slot the GPU left as it was)
        part_end();
        CHECK(guest[3 * 1024 + 10] == 5);
        CHECK(guest[3 * 1024 + 20] == (snapshot ? 7u : 0u));   // (without the snapshot: the mirror's empty slot over it)
        CHECK(guest[3 * 1024 + 900] == (known ? 0u : 4u));
        CHECK(filled == (snapshot ? 0 : 1));   // (a pinned page is never read from guest memory at the end)
        CHECK(pins_gone());
        if (!snapshot) continue;
        // the next part simulates B: the page was reported, so the shadow does not know it, and the snapshot has B from the sync
        part_begin(3 * 4096, 4096);
        gpu_starts();
        CHECK(mirror_start[3 * 1024 + 20] == 7);   // (synced)
        mirror[3 * 1024 + 20] = 8;   // (B moved)
        part_end();
        CHECK(guest[3 * 1024 + 20] == 8 && shadow[3 * 1024 + 20] == 8 && pins_gone());
    }
    // PR 1109's counter: the replay clears it in stream order (guest_cpu_write), the GPU clears it too and counts it back to last
    // frame's value. The page is not pinned: read at the end as before, and guest memory ends as the whole copy leaves it
    {
        reset();
        part_begin(4096, 4096);
        gpu_starts();
        const uint32_t cleared = 1;
        CHECK(guest_cpu_write(kLo + 4096 + 12, &cleared, 4) == 1);
        mirror[1024 + 3] = 0;   // (counted back to last frame's value)
        part_end();
        CHECK(guest[1024 + 3] == 0 && filled == 1 && pins_gone());
        // and a page only the game wrote, in the same part, stays pinned beside it
        reset();
        part_begin(4096, 2 * 4096);
        gpu_starts();
        CHECK(guest_cpu_write(kLo + 4096 + 12, &cleared, 4) == 1);
        game_write(2 * 1024 + 30, 9);
        part_end();
        CHECK(guest[1024 + 3] == 0 && guest[2 * 1024 + 30] == (snapshot ? 9u : 0u));
    }
    // more reports than the lists keep: the replay's pages among them, so none pinned
    if (snapshot) {
        reset();
        part_begin(0, 4096);
        gpu_starts();
        shadows_pin_writebacks();
        CHECK(shadow_is_pinned(pool(), 0));
        std::vector<uint32_t> many(1u << 20, 0x7fff0);
        for (int i = 0; i < 10; i++) shadow_written(many.data(), (int64_t)many.size());
        CHECK(shadow_written_all);
        shadows_unpin_replay();
        CHECK(!shadow_is_pinned(pool(), 0));
        shadow_pins_clear();
    }
    // a parallel replay: the pages pinned before its segments are copied ahead of them, and none is pinned after
    if (snapshot) {
        reset();
        game_write(5 * 1024 + 1, 2);
        part_begin(5 * 4096, 4096);
        gpu_starts();
        shadows_before_segments();   // (in the uploads ahead of every segment: the copies run there)
        CHECK(shadow[5 * 1024 + 1] == 2 && shadow_snaps.empty() && shadow_is_pinned(pool(), 5));
        writebacks.push_back({&pool(), 6 * 4096, 7 * 4096});   // (bound in a segment)
        par_active = true;
        shadows_pin_writebacks();
        par_active = false;
        writebacks.push_back({&pool(), 7 * 4096, 8 * 4096});   // (bound after the segments, in the same submission)
        shadows_pin_writebacks();
        CHECK(!shadow_is_pinned(pool(), 6) && !shadow_is_pinned(pool(), 7) && shadow_is_pinned(pool(), 5));
        writebacks.resize(1);
        mirror[5 * 1024 + 2] = 3;    // (a segment's work)
        game_write(5 * 1024 + 4, 6);
        part_end();
        CHECK(guest[5 * 1024 + 2] == 3 && guest[5 * 1024 + 4] == 6 && pins_gone());
        mirror_epoch++;
        writebacks.push_back({&pool(), 7 * 4096, 8 * 4096});   // (the next submission pins again)
        shadows_pin_writebacks();
        CHECK(shadow_is_pinned(pool(), 7));
        writebacks.clear();
        shadow_pins_clear();
    }
    // against the rule over random parts: in the part's range a word the GPU changed is the GPU's, any other is what guest memory
    // holds at the end (the CPU's words written meanwhile kept); outside it nothing changes
    if (snapshot) {
        reset();
        uint64_t seed = 0x9e3779b97f4a7c15ull;
        const auto rnd = [&](uint64_t n) { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return (seed >> 33) % n; };
        for (int step = 0; step < 4000; step++) {
            for (int k = 0; k < 3; k++) game_write(rnd(kWords), (uint32_t)rnd(3));   // (before the part: synced into the mirror)
            const uint64_t pg = rnd(kPages), n = 1 + rnd(kPages - pg), at = pg * 4096 + rnd(64) * 4, len = n * 4096 - at % 4096;
            part_begin(at, len);
            gpu_starts();
            for (int k = 0; k < 6; k++) mirror[(at + rnd(len)) / 4] = (uint32_t)rnd(3);   // (the GPU's work, in its range)
            for (int k = 0; k < 3; k++) game_write(rnd(kWords), (uint32_t)rnd(3));   // (meanwhile: not in this part's mirror)
            std::vector<uint32_t> want = guest;
            for (uint64_t w = at / 4; w < (at + len) / 4; w++)
                if (mirror[w] != mirror_start[w]) want[w] = mirror[w];
            part_end();
            CHECK(guest == want);
            CHECK(pins_gone());
        }
    }
    std::printf("PASS: the write-back diff %s the CPU's words where the GPU left them (RX_VK_WB_SNAPSHOT %s)\n",
                snapshot ? "keeps" : "loses", snapshot ? "on" : "=0");
}
'''
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / 'spawns.cpp'
        exe = Path(tmp) / 'spawns.exe'
        src.write_text(fixture, encoding='utf-8')
        subprocess.run([args.cxx, '-std=c++20', '-O1', '-Wall', '-Wextra', '-Werror', str(src), '-o', str(exe)], check=True)
        for value, arg in ((None, '1'), ('1', '1'), ('0', '0')):
            env = dict(os.environ)
            env.pop('RX_VK_WB_SNAPSHOT', None)
            if value is not None:
                env['RX_VK_WB_SNAPSHOT'] = value
            subprocess.run([str(exe), arg], env=env, check=True)


if __name__ == '__main__':
    main()
