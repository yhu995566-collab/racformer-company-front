#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace racformer {

// Fixed-capacity newest-first history. Snapshots retain shared ownership so a
// concurrent reset or a later ring overwrite cannot invalidate an inference
// that is already using the previous temporal window.
template <typename T, std::size_t Capacity>
class TemporalRing {
    static_assert(Capacity > 0, "TemporalRing capacity must be positive");

 public:
    using Entry = std::shared_ptr<const T>;
    using Window = std::array<Entry, Capacity>;

    bool empty() const noexcept { return size_ == 0; }
    std::size_t size() const noexcept { return size_; }

    const T& newest() const {
        if (empty()) throw std::logic_error("TemporalRing is empty");
        return *slots_[head_];
    }

    void push(T value) {
        const std::size_t next = empty() ? 0 : (head_ + Capacity - 1) % Capacity;
        slots_[next] = std::make_shared<const T>(std::move(value));
        head_ = next;
        if (size_ < Capacity) ++size_;
    }

    Window padded_window() const {
        if (empty()) throw std::logic_error("TemporalRing is empty");
        Window result{};
        for (std::size_t index = 0; index < size_; ++index) {
            result[index] = slots_[(head_ + index) % Capacity];
        }
        const Entry& oldest = result[size_ - 1];
        for (std::size_t index = size_; index < Capacity; ++index) {
            result[index] = oldest;
        }
        return result;
    }

    void clear() noexcept {
        for (Entry& entry : slots_) entry.reset();
        head_ = 0;
        size_ = 0;
    }

 private:
    std::array<Entry, Capacity> slots_{};
    std::size_t head_{};
    std::size_t size_{};
};

}  // namespace racformer
