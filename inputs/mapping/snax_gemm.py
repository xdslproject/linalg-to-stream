mapping = {
    "default": {  # default
        "spatial_mapping": {"PE_ARRAY_D1": ("d0", 8), "PE_ARRAY_D2": ("d1", 8), "PE_ARRAY_D3": ("d2", 8)},
        "temporal_ordering": [
            # Innermost loop
            # ("K", 8),
            # ("N", 8),
            # ("M", 8),
            # ("K", 8),
            # ("N", 8),
            # ("M", 8),
            # Outermost loop
        ],
        "core_allocation": 1,
        "memory_operand_links": {
            "O": "O",
            "W": "I2",
            "I": "I1",
        },
    },
}
