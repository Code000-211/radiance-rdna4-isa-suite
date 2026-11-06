; WMMA result D (v[8:15]) feeds the A operand of the next WMMA with nothing in between.
v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15]
v_wmma_f32_16x16x16_f16 v[16:23], v[8:11], v[4:7], v[16:23]
s_endpgm
