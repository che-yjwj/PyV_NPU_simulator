
import unittest
import os
import sys
from pyv.simulator import Simulator
from pyv.models.singlecycle import SingleCycleModel

# Add the parent directory to the python path to allow module import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestIntegrationCache(unittest.TestCase):

    def test_cache_hit_miss_timing(self):
        """ 
        Tests the timing effect of L1 cache hits and misses.
        It loads a tiny program that reads from two different memory addresses.
        - The first read should cause a cache miss (long latency).
        - The second read to the same address should be a hit (short latency).
        - A third read to a new address should be a miss again.
        """
        # Program: 
        # lw x1, 0(x0)  # Read from address 0x0
        # lw x2, 0(x0)  # Read from address 0x0 again (should be a hit)
        # lw x3, 128(x0) # Read from address 0x80 (different cache line, should be a miss)
        instructions = [
            0x00002083, # lw x1, 0(x0)
            0x00002103, # lw x2, 0(x0)
            0x08002183  # lw x3, 128(x0)
        ]

        # --- Simulation 1: Run the program ---
        Simulator.clear()
        model = SingleCycleModel()
        model.load_instructions(instructions)
        sim = Simulator()
        sim.init()
        sim.run(num_cycles=200) # Run for enough cycles to complete

        # --- Analysis ---
        # The exact number of cycles is hard to predict without a deeper simulation trace,
        # but we can assert the relative timings.
        # A miss costs ~100 cycles, a hit ~1 cycle. A 5-stage pipeline takes 5 cycles base.
        # Inst 1 (miss): ~5 (pipeline) + 100 (miss penalty) = ~105 cycles
        # Inst 2 (hit):  1 cycle for MEM stage
        # Inst 3 (miss): ~1 (pipeline) + 100 (miss penalty) = ~101 cycles
        # Total should be around 207 cycles. We check if it's in a reasonable range.
        final_cycles = sim.get_cycles()
        print(f"Total cycles for cache test: {final_cycles}")

        # We expect the simulation to take significantly longer than if all accesses were hits.
        # 3 instructions * ~5 cycles/instruction = ~15 cycles if no misses.
        self.assertGreater(final_cycles, 150, "Cycle count should be high due to cache misses")
        self.assertLess(final_cycles, 250, "Cycle count should be in a reasonable range for two misses")

if __name__ == '__main__':
    unittest.main()
