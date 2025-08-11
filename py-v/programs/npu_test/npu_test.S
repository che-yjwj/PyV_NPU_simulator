# npu_test.S
# Tests ring buffer, and status polling.

.section .text
.global _start

_start:
    # --- Descriptor 1: MatMul at 0x1000 ---
    lui     a0, %hi(0x1000)
    addi    a0, a0, %lo(0x1000)

    li      a1, 0           # opcode = 0 (MatMul)
    sw      a1, 0(a0)
    li      a1, 0           # flags = 0
    sw      a1, 4(a0)
    li      a1, 0x2000      # in0_addr
    sw      a1, 8(a0)
    li      a1, 0x3000      # in1_addr
    sw      a1, 16(a0)
    li      a1, 0x4000      # out_addr
    sw      a1, 24(a0)
    li      a1, 128         # tile_m
    sw      a1, 32(a0)
    li      a1, 128         # tile_n
    sw      a1, 36(a0)
    li      a1, 64          # tile_k
    sw      a1, 40(a0)
    li      a1, 1234        # ticket
    sw      a1, 60(a0)

    # --- Descriptor 2: GELU at 0x1040 ---
    addi    a0, a0, 64      # Move to next descriptor slot

    li      a1, 1           # opcode = 1 (GELU)
    sw      a1, 0(a0)
    li      a1, 0           # flags = 0
    sw      a1, 4(a0)
    li      a1, 0x5000      # in0_addr
    sw      a1, 8(a0)
    li      a1, 0x6000      # out_addr
    sw      a1, 24(a0)
    li      a1, 128         # tile_m (used as size)
    sw      a1, 32(a0)
    li      a1, 4321        # ticket
    sw      a1, 60(a0)

    # --- MMIO Setup ---

    # 1. Set QUEUE_BASE to 0x1000
    lui     a2, %hi(0x40000000)         # QUEUE_BASE_LO address
    addi    a2, a2, %lo(0x40000000)
    li      a1, 0x1000
    sw      a1, 0(a2)                   # Write to QUEUE_BASE_LO

    # 2. Set QUEUE_TAIL to 2
    lui     a2, %hi(0x40000018)         # QUEUE_TAIL_LO address
    addi    a2, a2, %lo(0x40000018)
    li      a1, 2                       # new tail is 2
    sw      a1, 0(a2)                   # Write to QUEUE_TAIL_LO

    # 3. Ring the DOORBELL with the new tail value
    lui     a2, %hi(0x40000020)         # DOORBELL address
    addi    a2, a2, %lo(0x40000020)
    li      a1, 2                       # value written is the new tail
    sw      a1, 0(a2)                   # Ring the doorbell

    # 4. Poll IRQ_STATUS for completion
    lui     a3, %hi(0x40000024)         # IRQ_STATUS address
    addi    a3, a3, %lo(0x40000024)

poll_loop:
    lw      a4, 0(a3)                   # Read IRQ_STATUS
    andi    a5, a4, 1                   # Check if bit 0 is set
    beq     a5, zero, poll_loop         # If not, loop again

    # 5. Acknowledge and clear IRQ by writing 1 back
    li      a5, 1
    sw      a5, 0(a3)

    # --- Done ---
done_loop:
    j       done_loop