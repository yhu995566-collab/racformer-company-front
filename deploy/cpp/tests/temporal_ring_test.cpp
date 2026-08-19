#include "temporal_ring.hpp"

#include <cassert>

namespace {
struct Frame { int id; };
}

int main() {
    racformer::TemporalRing<Frame, 4> ring;
    assert(ring.empty());

    ring.push(Frame{1});
    auto startup = ring.padded_window();
    for (const auto& frame : startup) assert(frame && frame->id == 1);

    ring.push(Frame{2});
    ring.push(Frame{3});
    ring.push(Frame{4});
    auto full = ring.padded_window();
    assert(full[0]->id == 4);
    assert(full[1]->id == 3);
    assert(full[2]->id == 2);
    assert(full[3]->id == 1);

    // A snapshot must stay alive and unchanged after its slot is overwritten.
    ring.push(Frame{5});
    auto shifted = ring.padded_window();
    assert(shifted[0]->id == 5);
    assert(shifted[3]->id == 2);
    assert(full[3]->id == 1);

    ring.clear();
    assert(ring.empty());
    // Clear must not invalidate an in-flight snapshot.
    assert(shifted[0]->id == 5);
    return 0;
}
