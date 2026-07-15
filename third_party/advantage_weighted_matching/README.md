# Advantage-Weighted Matching subset

This directory vendors the SD3 training path from
`scxue/advantage_weighted_matching` for the SpectraReward reproduction. The
upstream project is licensed under Apache-2.0; its license is retained here.
The vendored source baseline is commit
`15f91b8596168bc2399824eea18c0e1d07df7781`.

Local changes are limited to AlphaGRPO20k loading, the SGLang SpectraReward
adapter, exact optimizer-step stopping, and resumable cluster checkpoints.
