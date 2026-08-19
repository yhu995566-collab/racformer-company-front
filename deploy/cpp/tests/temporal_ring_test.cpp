#include "temporal_ring.hpp"

#include <iostream>

namespace {
struct Frame { int id; };

bool check(bool condition, const char* message) {
    if (condition) return true;
    std::cerr << "temporal ring check failed: " << message << '\n';
    return false;
}
}

int main() {
    racformer::TemporalRing<Frame, 4> ring;
    if (!check(ring.empty(), "new ring must be empty")) return 1;

    ring.push(Frame{1});
    auto startup = ring.padded_window();
    for (const auto& frame : startup) {
        if (!check(frame && frame->id == 1,
                   "startup padding must repeat the oldest frame")) return 1;
    }

    ring.push(Frame{2});
    ring.push(Frame{3});
    ring.push(Frame{4});
    auto full = ring.padded_window();
    if (!check(full[0]->id == 4, "newest frame must be first")) return 1;
    if (!check(full[1]->id == 3, "second frame order")) return 1;
    if (!check(full[2]->id == 2, "third frame order")) return 1;
    if (!check(full[3]->id == 1, "oldest frame must be last")) return 1;

    // A snapshot must stay alive and unchanged after its slot is overwritten.
    ring.push(Frame{5});
    auto shifted = ring.padded_window();
    if (!check(shifted[0]->id == 5, "overwritten window newest frame")) return 1;
    if (!check(shifted[3]->id == 2, "overwritten window oldest frame")) return 1;
    if (!check(full[3]->id == 1, "old snapshot must survive overwrite")) return 1;

    ring.clear();
    if (!check(ring.empty(), "clear must empty the ring")) return 1;
    // Clear must not invalidate an in-flight snapshot.
    if (!check(shifted[0]->id == 5, "snapshot must survive clear")) return 1;
    return 0;
}
