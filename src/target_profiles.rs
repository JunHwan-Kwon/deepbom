use super::*;

pub(super) fn is_delegate_break_suspect(name: &str) -> bool {
    matches!(
        name,
        "BATCH_MATMUL"
            | "EXPAND_DIMS"
            | "GATHER"
            | "MEAN"
            | "PACK"
            | "PAD"
            | "REDUCE_MAX"
            | "REDUCE_PROD"
            | "RESHAPE"
            | "RESIZE_BILINEAR"
            | "RSQRT"
            | "SHAPE"
            | "SOFTMAX"
            | "SQUARED_DIFFERENCE"
            | "STRIDED_SLICE"
            | "SUB"
            | "TRANSPOSE"
    )
}

fn cortex_a55_profile(id: &str, label: &str, l1_data_bytes: usize) -> TargetProfile {
    TargetProfile {
        id: id.to_string(),
        label: label.to_string(),
        profile_sha256: String::new(),
            derived_from: None,
            compute_utilization_by_kernel_class: BTreeMap::new(),
            compute_utilization_entries: Vec::new(),
        architecture: "AArch64 ARMv8.2-A Cortex-A55, in-order, NEON".to_string(),
        core_count_min: None,
        core_count_max: None,
        performance_reference_core_count: None,
        l1_data_bytes,
        l2_bytes: 128 * 1024,
        l2_capacity_scope: "private_per_core".to_string(),
        cache_assumption: format!(
            "Configurable Cortex-A55 planning profile: selected L1D {} KiB per core and private L2 128 KiB; Arm documents L1D 8-64 KiB and private L2 64-256 KiB. Not observed from a device.",
            l1_data_bytes / 1024
        ),
        cache_source_url: "https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Cortex-A%20R%20M%20datasheets/Arm%20Cortex-A%20Comparison%20Table_v4.pdf".to_string(),
        hardware_spec: None,
        performance_model_evidence_class: "HEURISTIC".to_string(),
        performance_model_assumption: "Uncalibrated Cortex-A55 planning constants for static comparison; bandwidth, sustained XNNPACK throughput, packing bandwidth, and boundary overhead are not established by the cache source.".to_string(),
        simd_width_bits: 128,
        fp32_lanes: 4,
        fp16_lanes: 8,
        int8_lanes: 16,
        in_order: true,
        dot_product: true,
        sve2: false,
        xnnpack_kernel_family: "NEON dot-product, in-order tuned".to_string(),
        effective_memory_bandwidth_gbps: 12.5,
        effective_peak_gops: 300.0,
        compute_utilization_factor: 1.0,
        ridge_point_ops_per_byte: 24.0,
        memory_bound_intensity: 6.0,
        compute_bound_intensity: 24.0,
        fp32_compute_factor: 4.0,
        int8_speedup_estimate: 4.0,
        channel_alignment_multiple: 16,
        weight_packing_bandwidth_gbps: 7.0,
        chain_break_overhead_us_low: 35.0,
        chain_break_overhead_us_high: 120.0,
    }
}

pub(super) fn is_cortex_a55_profile(target_id: &str) -> bool {
    target_id == "android_mid_a55" || target_id.starts_with("android_mid_a55_l1_")
}

pub(super) fn all_target_profiles() -> Vec<TargetProfile> {
    let mut profiles = vec![
        TargetProfile {
            id: "rpi4_a72".to_string(),
            label: "RPi4 / Cortex-A72".to_string(),
            profile_sha256: String::new(),
            derived_from: None,
            compute_utilization_by_kernel_class: BTreeMap::new(),
            compute_utilization_entries: Vec::new(),
            architecture: "ARMv8-A Cortex-A72, out-of-order, NEON".to_string(),
            core_count_min: Some(4),
            core_count_max: Some(4),
            performance_reference_core_count: Some(4),
            l1_data_bytes: 32 * 1024,
            l2_bytes: 1024 * 1024,
            l2_capacity_scope: "shared_cluster_unbound_concurrency".to_string(),
            cache_assumption: "RPi4 BCM2711 implementation: 32 KiB L1D per core and 1 MiB L2; documented device configuration, not browser-observed.".to_string(),
            cache_source_url: "https://www.raspberrypi.com/documentation/computers/processors.html".to_string(),
            hardware_spec: Some(TargetHardwareSpec {
                evidence_class: "SOURCE_BACKED_CORE_AND_PRODUCT".to_string(),
                scope: "Cortex-A72 core cache geometry plus the BCM2711 1 MiB shared-L2 selection used by this Raspberry Pi 4 profile; no cache topology was observed from the executing browser.".to_string(),
                configuration_context: "The Cortex-A72 TRM fixes L1I at 48 KiB and L1D at 32 KiB; L2 is implementation-configurable at 512 KiB, 1 MiB, 2 MiB, or 4 MiB. This profile binds the 1 MiB BCM2711 implementation rather than treating 1 MiB as a universal Cortex-A72 property.".to_string(),
                core_configuration: "Cortex-A72 supports 1-4 cores; BCM2711 profile binds four cores sharing L2.".to_string(),
                max_clock_mhz: None,
                l1_instruction_bytes: 48 * 1024,
                l1_data_bytes: 32 * 1024,
                l1_instruction_ways: 3,
                l1_data_ways: 2,
                l1_line_bytes: 64,
                l2_bytes: 1024 * 1024,
                l2_ways: 16,
                l2_line_bytes: 64,
                advanced_simd: true,
                fp16_vector_arithmetic: false,
                sources: vec![
                    TargetHardwareSource {
                        document: "Arm Cortex-A72 MPCore Processor Technical Reference Manual".to_string(),
                        revision: "r0p3; ARM 100095_0003_06_en".to_string(),
                        pages: "1-15, 1-17, 1-19, 2-26, 2-27, 6-285, 7-299".to_string(),
                        sha256: "47f52c93806507962c2cd0b77991d7aebacfaae5734ac617fc45a5babdff738f".to_string(),
                        url: String::new(),
                        scope: "ARMv8-A and Advanced SIMD; fixed L1 capacities; L1/L2 line size and associativity; supported core-count and L2-size configuration ranges.".to_string(),
                    },
                    TargetHardwareSource {
                        document: "Raspberry Pi processor documentation".to_string(),
                        revision: "live product documentation; content digest not embedded".to_string(),
                        pages: "BCM2711 processor/cache description".to_string(),
                        sha256: String::new(),
                        url: "https://www.raspberrypi.com/documentation/computers/processors.html".to_string(),
                        scope: "BCM2711 product binding, including the selected 1 MiB shared L2 implementation.".to_string(),
                    },
                ],
            }),
            performance_model_evidence_class: "HEURISTIC".to_string(),
            performance_model_assumption: "Cache geometry, clock, and the LPDDR4-2400 interface are source-backed (Cortex-A72 TRM; Raspberry Pi BCM2711 documentation). effective_memory_bandwidth_gbps carries the 9.6 GB/s interface peak that 2400 MT/s over 32 bits implies, and effective_peak_gops carries a 192 GOPS issue ceiling; both are maxima chosen so the roofline stays a lower bound on time, not throughput a kernel will reach. The Advanced SIMD pipe count behind the 192 figure is the one input not taken from a manual. Sustained XNNPACK throughput, packing bandwidth, and delegate-boundary overhead remain uncalibrated; fit compute_utilization_by_kernel_class against benchmark_model before treating any of this as latency.".to_string(),
            simd_width_bits: 128,
            fp32_lanes: 4,
            fp16_lanes: 8,
            int8_lanes: 16,
            in_order: false,
            dot_product: false,
            sve2: false,
            xnnpack_kernel_family: "NEON 128-bit".to_string(),
            // BCM2711 accesses LPDDR4-2400 over a 32-bit interface, so the
            // interface peak is 2400 MT/s x 4 B = 9.6 GB/s. The previous 25 GB/s
            // was above what the documented interface can carry, which made the
            // memory term of the roofline meaninglessly loose.
            effective_memory_bandwidth_gbps: 9.6,
            // 4 cores x 1.5 GHz x 2 Advanced SIMD pipes x SMLAL (8 widening
            // INT8 MACs, counted as 16 ops) = 192 GOPS. Cortex-A72 has no
            // Armv8.2 dot-product instruction, which is why the per-cycle
            // figure is a quarter of the A76 profile's. The pipe count is the
            // one input not taken from a manual.
            effective_peak_gops: 192.0,
            compute_utilization_factor: 1.0,
            // 192 GOPS / 9.6 GB/s = 20.0 ops per byte. The previous constants
            // produced the same ridge because both were inflated by the same
            // factor, so the bound classification was right while the absolute
            // time was not.
            ridge_point_ops_per_byte: 20.0,
            memory_bound_intensity: 5.9,
            compute_bound_intensity: 20.0,
            // A72 has 16 INT8 lanes vs 4 FP32 lanes in 128-bit NEON; no dot-product extension.
            // Effective ratio ≈ 4× (narrower SIMD packing for INT8 via VMLAL vs VMLA).
            fp32_compute_factor: 4.0,
            int8_speedup_estimate: 1.7,
            channel_alignment_multiple: 8,
            weight_packing_bandwidth_gbps: 12.0,
            chain_break_overhead_us_low: 18.0,
            chain_break_overhead_us_high: 55.0,
        },
        TargetProfile {
            id: "rpi5_a76".to_string(),
            label: "RPi5 / Cortex-A76".to_string(),
            profile_sha256: String::new(),
            derived_from: None,
            compute_utilization_by_kernel_class: BTreeMap::new(),
            compute_utilization_entries: Vec::new(),
            architecture: "AArch64 Armv8.2-A Cortex-A76, out-of-order, NEON; optional Armv8.4-A dot-product".to_string(),
            core_count_min: Some(4),
            core_count_max: Some(4),
            performance_reference_core_count: Some(4),
            l1_data_bytes: 64 * 1024,
            l2_bytes: 512 * 1024,
            l2_capacity_scope: "private_per_core".to_string(),
            cache_assumption: "Cortex-A76 fixes L1I and L1D at 64 KiB each and offers L2 at 128 KiB, 256 KiB, or 512 KiB; BCM2712 selects the 512 KiB private per-core option and adds a 2 MiB shared L3 that this planning model does not represent. Documented core and product configuration, not device-observed.".to_string(),
            cache_source_url: "https://www.raspberrypi.com/documentation/computers/processors.html".to_string(),
            hardware_spec: Some(TargetHardwareSpec {
                evidence_class: "SOURCE_BACKED_CORE_AND_PRODUCT".to_string(),
                scope: "Cortex-A76 core cache geometry and SIMD feature set, plus the BCM2712 512 KiB L2 selection and 2.4 GHz clock used by this Raspberry Pi 5 profile. The 2 MiB shared L3 that BCM2712 adds has no field in this planning model and is not represented here.".to_string(),
                configuration_context: "The Cortex-A76 TRM fixes both L1 caches at 64 KiB, 4-way, with 64-byte lines, and makes L2 configurable at 128 KiB, 256 KiB, or 512 KiB, 8-way. This profile binds the 512 KiB BCM2712 selection rather than treating it as a universal Cortex-A76 property.".to_string(),
                core_configuration: "BCM2712 profile binds four Cortex-A76 cores, each with a private L2, behind a 2 MiB shared L3.".to_string(),
                max_clock_mhz: Some(2400),
                l1_instruction_bytes: 64 * 1024,
                l1_data_bytes: 64 * 1024,
                l1_instruction_ways: 4,
                l1_data_ways: 4,
                l1_line_bytes: 64,
                l2_bytes: 512 * 1024,
                l2_ways: 8,
                l2_line_bytes: 64,
                advanced_simd: true,
                // ID_AA64PFR0_EL1 reports "Advanced SIMD, including
                // Half-precision support, is implemented", so unlike the A72
                // profile this core has FP16 arithmetic and not only conversion.
                fp16_vector_arithmetic: true,
                sources: vec![
                    TargetHardwareSource {
                        document: "Arm Cortex-A76 Core Technical Reference Manual".to_string(),
                        revision: "r3p0; 100798_0300_00_en".to_string(),
                        pages: "A2.1 instruction fetch and load/store unit; A6 L2 memory system; ID_AA64ISAR0_EL1 and ID_AA64PFR0_EL1 register descriptions".to_string(),
                        sha256: "2be478af4f33c12eacd854c4d354373c647dc6c100d43fa44aedbb112aa8c629".to_string(),
                        url: "https://developer.arm.com/documentation/100798/0300/".to_string(),
                        scope: "Fixed 64 KiB 4-way L1I and L1D with 64-byte lines; 8-way L2 configurable at 128/256/512 KiB with 64-byte lines; Advanced SIMD with half-precision arithmetic; optional Armv8.4-A SDOT/UDOT.".to_string(),
                    },
                    TargetHardwareSource {
                        document: "Raspberry Pi processor documentation".to_string(),
                        revision: "live product documentation; content digest not embedded".to_string(),
                        pages: "BCM2712 processor/cache description".to_string(),
                        sha256: String::new(),
                        url: "https://www.raspberrypi.com/documentation/computers/processors.html".to_string(),
                        scope: "BCM2712 product binding: quad Cortex-A76 at up to 2.4 GHz, 64 KiB I and D caches, 512 KiB per-core L2, 2 MiB shared L3, and a 32-bit LPDDR4X interface documented at up to 17 GB/s.".to_string(),
                    },
                ],
            }),
            performance_model_evidence_class: "HEURISTIC".to_string(),
            performance_model_assumption: "Cache geometry, clock, ISA features, and the documented 17 GB/s LPDDR4X peak are source-backed (Cortex-A76 TRM; Raspberry Pi BCM2712 documentation). The throughput and bandwidth constants below are planning values, not measurements: effective_peak_gops assumes two 128-bit Advanced SIMD pipelines issuing UDOT at 2.4 GHz across four cores, and effective_memory_bandwidth_gbps takes the documented interface peak, because a roofline term below the hardware maximum would stop being a lower bound. Calibrate compute_utilization_by_kernel_class against benchmark_model before treating any of this as latency.".to_string(),
            simd_width_bits: 128,
            fp32_lanes: 4,
            fp16_lanes: 8,
            int8_lanes: 16,
            in_order: false,
            // The TRM states the core "optionally implements the SDOT and UDOT
            // instructions" from the Armv8.4-A extension, so this is an
            // implementation choice rather than a guaranteed core property. The
            // profile binds the BCM2712 assumption; on a running device it is
            // confirmed by the asimddp flag in /proc/cpuinfo.
            dot_product: true,
            sve2: false,
            xnnpack_kernel_family: "NEON dot-product, out-of-order tuned".to_string(),
            // A roofline term is only a valid lower bound on time while its
            // constant is at least what the hardware can do, so this carries the
            // documented interface peak rather than a streaming fraction. The
            // gap to achievable throughput belongs in
            // compute_utilization_by_kernel_class, which is measured.
            // BCM2712: 32-bit LPDDR4X, documented up to 17 GB/s.
            effective_memory_bandwidth_gbps: 17.0,
            // 4 cores x 2.4 GHz x 2 Advanced SIMD pipes x UDOT (16 INT8 MACs,
            // counted as 32 ops) = 614 GOPS. The pipe count is the one input
            // here not taken from the manual, so this is a planning ceiling.
            effective_peak_gops: 614.4,
            compute_utilization_factor: 1.0,
            // 614.4 GOPS / 17 GB/s = 36.1 ops per byte.
            ridge_point_ops_per_byte: 36.1,
            memory_bound_intensity: 10.6,
            compute_bound_intensity: 36.1,
            // 16 INT8 lanes against 4 FP32 lanes in 128-bit NEON, but the
            // dot-product extension recovers most of the packing gap, so the
            // effective ratio is lower than the A72 profile's 4x.
            fp32_compute_factor: 2.5,
            int8_speedup_estimate: 2.2,
            channel_alignment_multiple: 8,
            weight_packing_bandwidth_gbps: 8.0,
            chain_break_overhead_us_low: 14.0,
            chain_break_overhead_us_high: 45.0,
        },
        TargetProfile {
            id: "android_mid_a55".to_string(),
            label: "Android mid-range / Cortex-A55 (L1D 32 KiB default)".to_string(),
            profile_sha256: String::new(),
            derived_from: None,
            compute_utilization_by_kernel_class: BTreeMap::new(),
            compute_utilization_entries: Vec::new(),
            architecture: "AArch64 ARMv8.2-A Cortex-A55, in-order, NEON".to_string(),
            core_count_min: None,
            core_count_max: None,
            performance_reference_core_count: None,
            l1_data_bytes: 32 * 1024,
            l2_bytes: 128 * 1024,
            l2_capacity_scope: "private_per_core".to_string(),
            cache_assumption: "Configurable Cortex-A55 planning profile: selected L1D 32 KiB per core and private L2 128 KiB; Arm documents L1D 8-64 KiB and private L2 64-256 KiB. Not observed from a device.".to_string(),
            cache_source_url: "https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Cortex-A%20R%20M%20datasheets/Arm%20Cortex-A%20Comparison%20Table_v4.pdf".to_string(),
            hardware_spec: None,
            performance_model_evidence_class: "HEURISTIC".to_string(),
            performance_model_assumption: "Uncalibrated Cortex-A55 planning constants for static comparison; bandwidth, sustained XNNPACK throughput, packing bandwidth, and boundary overhead are not established by the cache source.".to_string(),
            simd_width_bits: 128,
            fp32_lanes: 4,
            fp16_lanes: 8,
            int8_lanes: 16,
            in_order: true,
            dot_product: true,
            sve2: false,
            xnnpack_kernel_family: "NEON dot-product, in-order tuned".to_string(),
            effective_memory_bandwidth_gbps: 12.5,
            effective_peak_gops: 300.0,
            compute_utilization_factor: 1.0,
            ridge_point_ops_per_byte: 24.0,
            memory_bound_intensity: 6.0,
            compute_bound_intensity: 24.0,
            // A55 with dot-product: UDOT/SDOT gives 4× INT8 throughput vs FP32 FMLA.
            fp32_compute_factor: 4.0,
            int8_speedup_estimate: 4.0,
            channel_alignment_multiple: 16,
            weight_packing_bandwidth_gbps: 7.0,
            chain_break_overhead_us_low: 35.0,
            chain_break_overhead_us_high: 120.0,
        },
        cortex_a55_profile(
            "android_mid_a55_l1_16k",
            "Android cost / Cortex-A55 (L1D 16 KiB)",
            16 * 1024,
        ),
        cortex_a55_profile(
            "android_mid_a55_l1_64k",
            "Android Cortex-A55 (L1D 64 KiB option)",
            64 * 1024,
        ),
        TargetProfile {
            id: "zynq_ultrascale_plus_a53".to_string(),
            label: "Zynq UltraScale+ MPSoC / Cortex-A53 (DS891)".to_string(),
            profile_sha256: String::new(),
            derived_from: None,
            compute_utilization_by_kernel_class: BTreeMap::new(),
            compute_utilization_entries: Vec::new(),
            architecture: "Armv8-A Cortex-A53, in-order, NEON; Zynq UltraScale+ MPSoC APU".to_string(),
            core_count_min: Some(2),
            core_count_max: Some(4),
            performance_reference_core_count: Some(4),
            l1_data_bytes: 32 * 1024,
            l2_bytes: 1024 * 1024,
            l2_capacity_scope: "shared_cluster_unbound_concurrency".to_string(),
            cache_assumption: "Product-documented Zynq UltraScale+ APU cache configuration: 32 KiB L1I and 32 KiB L1D per Cortex-A53 core, plus 1 MiB unified shared L2. CG devices are dual-core; EG/EV devices are quad-core. The selected cache denominator is source-backed, not browser-observed.".to_string(),
            cache_source_url: "https://docs.amd.com/v/u/en-US/ds891-zynq-ultrascale-plus-overview".to_string(),
            hardware_spec: Some(TargetHardwareSpec {
                evidence_class: "SOURCE_BACKED_PRODUCT".to_string(),
                scope: "Zynq UltraScale+ MPSoC application-processing unit. Cache capacity and product core count come from AMD DS891; cache line size and associativity are cross-checked against the Cortex-A53 TRM.".to_string(),
                configuration_context: "Cortex-A53 permits independently configured 8/16/32/64 KiB L1I and L1D plus optional 128/256/512/1024/2048 KiB L2. AMD DS891 resolves those options for Zynq UltraScale+ to 32 KiB/32 KiB L1 and 1 MiB L2.".to_string(),
                core_configuration: "CG: dual-core Cortex-A53; EG/EV: quad-core Cortex-A53; common cache capacities and NEON support.".to_string(),
                max_clock_mhz: Some(1500),
                l1_instruction_bytes: 32 * 1024,
                l1_data_bytes: 32 * 1024,
                l1_instruction_ways: 2,
                l1_data_ways: 4,
                l1_line_bytes: 64,
                l2_bytes: 1024 * 1024,
                l2_ways: 16,
                l2_line_bytes: 64,
                advanced_simd: true,
                fp16_vector_arithmetic: false,
                sources: vec![
                    TargetHardwareSource {
                        document: "AMD Zynq UltraScale+ MPSoC Data Sheet: Overview".to_string(),
                        revision: "DS891 v1.11.1; March 18, 2025".to_string(),
                        pages: "4, 6, 8, 12".to_string(),
                        sha256: "1badf7142690c573987f3eacd788620ff8a8392425f13124f928aaed152265e9".to_string(),
                        url: "https://docs.amd.com/v/u/en-US/ds891-zynq-ultrascale-plus-overview".to_string(),
                        scope: "CG/EG/EV core counts; 32 KiB/32 KiB L1; 1 MiB L2; up-to-1.5 GHz operating target; NEON and single/double-precision floating point.".to_string(),
                    },
                    TargetHardwareSource {
                        document: "Arm Cortex-A53 MPCore Processor Technical Reference Manual".to_string(),
                        revision: "r0p2; ARM DDI 0500D (ID021414)".to_string(),
                        pages: "1-7, 6-2, 7-2".to_string(),
                        sha256: "52b19d733bdacfbd1cffd108b277bfbc115839aab0a9f5d51f43b6dfa7c33369".to_string(),
                        url: String::new(),
                        scope: "Implementation-option ranges; 64-byte L1/L2 lines; 2-way L1I, 4-way L1D, and 16-way L2 organization; optional Advanced SIMD and floating point.".to_string(),
                    },
                ],
            }),
            performance_model_evidence_class: "HEURISTIC".to_string(),
            performance_model_assumption: "Cache and ISA facts are source-backed, but DS891 does not specify sustained application-memory bandwidth, XNNPACK INT8 throughput, packing bandwidth, thread count, governor, or delegate overhead. The conservative constants below are static prioritization assumptions and must not be reported as measured Zynq latency.".to_string(),
            simd_width_bits: 128,
            fp32_lanes: 4,
            fp16_lanes: 8,
            int8_lanes: 16,
            in_order: true,
            dot_product: false,
            sve2: false,
            xnnpack_kernel_family: "NEON 128-bit; no Armv8.2 dot-product".to_string(),
            // DS891 documents 32-bit or 64-bit DDR4 at up to 2400 Mb/s in the
            // -1 speed grade, so the widest supported interface carries
            // 2400 MT/s x 8 B = 19.2 GB/s. The previous 8.0 was below even the
            // 32-bit configuration, which would push the roofline above times
            // the hardware can actually achieve.
            effective_memory_bandwidth_gbps: 19.2,
            // 4 cores x 1.5 GHz x SMLAL (8 widening INT8 MACs, 16 ops) on the
            // A53 64-bit NEON datapath, which needs two cycles per 128-bit
            // operation = 48 GOPS. Cortex-A53 is Armv8.0 and has no
            // dot-product instruction.
            effective_peak_gops: 48.0,
            compute_utilization_factor: 1.0,
            // 48 GOPS / 19.2 GB/s = 2.5 ops per byte.
            ridge_point_ops_per_byte: 2.5,
            memory_bound_intensity: 0.7,
            compute_bound_intensity: 2.5,
            fp32_compute_factor: 2.0,
            int8_speedup_estimate: 1.5,
            channel_alignment_multiple: 8,
            weight_packing_bandwidth_gbps: 4.0,
            chain_break_overhead_us_low: 45.0,
            chain_break_overhead_us_high: 150.0,
        },
        TargetProfile {
            id: "android_flagship_x3_a715".to_string(),
            label: "Illustrative X3/A715 planning profile".to_string(),
            profile_sha256: String::new(),
            derived_from: None,
            compute_utilization_by_kernel_class: BTreeMap::new(),
            compute_utilization_entries: Vec::new(),
            architecture:
                "Illustrative ARMv9 Cortex-X3/A715-class planning profile; not a measured device specification"
                    .to_string(),
            core_count_min: None,
            core_count_max: None,
            performance_reference_core_count: None,
            l1_data_bytes: 64 * 1024,
            l2_bytes: 1024 * 1024,
            l2_capacity_scope: "mixed_core_profile_unbound".to_string(),
            cache_assumption: "Illustrative 64 KiB L1D / 1 MiB L2 planning values for a mixed Cortex-X3/A715-class label; not one identified SoC cache topology and not device-observed.".to_string(),
            cache_source_url: "https://developer.arm.com/community/arm-community-blogs/b/announcements/posts/compute-performance-unleashed".to_string(),
            hardware_spec: None,
            performance_model_evidence_class: "HEURISTIC".to_string(),
            performance_model_assumption: "Illustrative mixed-core-class planning constants; not a source-bound SoC or measured device profile.".to_string(),
            simd_width_bits: 256,
            fp32_lanes: 8,
            fp16_lanes: 16,
            int8_lanes: 32,
            in_order: false,
            dot_product: true,
            sve2: true,
            xnnpack_kernel_family: "SVE2 or NEON dot-product".to_string(),
            effective_memory_bandwidth_gbps: 35.0,
            effective_peak_gops: 5600.0,
            compute_utilization_factor: 1.0,
            ridge_point_ops_per_byte: 160.0,
            memory_bound_intensity: 32.0,
            compute_bound_intensity: 96.0,
            // SVE2 + dot-product: UDOT on 256-bit SVE2 gives ~4× INT8 over FP32 FMLA.
            fp32_compute_factor: 4.0,
            int8_speedup_estimate: 4.0,
            channel_alignment_multiple: 16,
            weight_packing_bandwidth_gbps: 18.0,
            chain_break_overhead_us_low: 20.0,
            chain_break_overhead_us_high: 70.0,
        },
        TargetProfile {
            id: "x86_avx2".to_string(),
            label: "x86 / AVX2".to_string(),
            profile_sha256: String::new(),
            derived_from: None,
            compute_utilization_by_kernel_class: BTreeMap::new(),
            compute_utilization_entries: Vec::new(),
            architecture: "x86-64 AVX2".to_string(),
            core_count_min: None,
            core_count_max: None,
            performance_reference_core_count: None,
            l1_data_bytes: 32 * 1024,
            l2_bytes: 256 * 1024,
            l2_capacity_scope: "private_per_core_planning_default".to_string(),
            cache_assumption: "Conservative x86 planning defaults; AVX2 identifies an ISA feature, not a cache topology. Replace with CPUID/OS-observed cache data for deployment claims.".to_string(),
            cache_source_url: String::new(),
            hardware_spec: None,
            performance_model_evidence_class: "HEURISTIC".to_string(),
            performance_model_assumption: "Generic AVX2 planning constants; AVX2 alone does not identify cache topology, sustained throughput, memory bandwidth, or runtime configuration.".to_string(),
            simd_width_bits: 256,
            fp32_lanes: 8,
            fp16_lanes: 8,
            int8_lanes: 32,
            in_order: false,
            dot_product: false,
            sve2: false,
            xnnpack_kernel_family: "AVX2".to_string(),
            effective_memory_bandwidth_gbps: 35.0,
            effective_peak_gops: 1680.0,
            compute_utilization_factor: 1.0,
            ridge_point_ops_per_byte: 48.0,
            memory_bound_intensity: 14.0,
            compute_bound_intensity: 48.0,
            // AVX2: 32 INT8 lanes vs 8 FP32 lanes (4× hardware), but XNNPACK INT8 uses
            // VPMADDUBSW chains (~2× effective over FP32 FMA in practice; no VNNI).
            fp32_compute_factor: 2.0,
            int8_speedup_estimate: 1.8,
            channel_alignment_multiple: 8,
            weight_packing_bandwidth_gbps: 18.0,
            chain_break_overhead_us_low: 12.0,
            chain_break_overhead_us_high: 40.0,
        },
        TargetProfile {
            id: "x86_sse4".to_string(),
            label: "x86 / SSE4".to_string(),
            profile_sha256: String::new(),
            derived_from: None,
            compute_utilization_by_kernel_class: BTreeMap::new(),
            compute_utilization_entries: Vec::new(),
            architecture: "x86-64 SSE4.x".to_string(),
            core_count_min: None,
            core_count_max: None,
            performance_reference_core_count: None,
            l1_data_bytes: 32 * 1024,
            l2_bytes: 256 * 1024,
            l2_capacity_scope: "private_per_core_planning_default".to_string(),
            cache_assumption: "Conservative x86 planning defaults; SSE4 identifies an ISA feature, not a cache topology. Replace with CPUID/OS-observed cache data for deployment claims.".to_string(),
            cache_source_url: String::new(),
            hardware_spec: None,
            performance_model_evidence_class: "HEURISTIC".to_string(),
            performance_model_assumption: "Generic SSE4 planning constants; SSE4 alone does not identify cache topology, sustained throughput, memory bandwidth, or runtime configuration.".to_string(),
            simd_width_bits: 128,
            fp32_lanes: 4,
            fp16_lanes: 4,
            int8_lanes: 16,
            in_order: false,
            dot_product: false,
            sve2: false,
            xnnpack_kernel_family: "SSE4 128-bit".to_string(),
            effective_memory_bandwidth_gbps: 22.0,
            effective_peak_gops: 528.0,
            compute_utilization_factor: 1.0,
            ridge_point_ops_per_byte: 24.0,
            memory_bound_intensity: 7.0,
            compute_bound_intensity: 24.0,
            // SSE4: 16 INT8 vs 4 FP32 (4× hardware), ~2× effective (no VNNI on this path).
            fp32_compute_factor: 2.0,
            int8_speedup_estimate: 1.35,
            channel_alignment_multiple: 8,
            weight_packing_bandwidth_gbps: 10.0,
            chain_break_overhead_us_low: 18.0,
            chain_break_overhead_us_high: 65.0,
        },
        TargetProfile {
            id: "wasm_simd".to_string(),
            label: "Browser / WASM SIMD".to_string(),
            profile_sha256: String::new(),
            derived_from: None,
            compute_utilization_by_kernel_class: BTreeMap::new(),
            compute_utilization_entries: Vec::new(),
            architecture: "WASM SIMD 128-bit browser runtime".to_string(),
            core_count_min: None,
            core_count_max: None,
            performance_reference_core_count: None,
            l1_data_bytes: 32 * 1024,
            l2_bytes: 256 * 1024,
            l2_capacity_scope: "host_dependent_unbound".to_string(),
            cache_assumption: "Conservative browser-host proxy; WASM SIMD does not expose host cache topology. Replace with a bound native host profile for deployment claims.".to_string(),
            cache_source_url: String::new(),
            hardware_spec: None,
            performance_model_evidence_class: "HEURISTIC".to_string(),
            performance_model_assumption: "Browser-host proxy constants; WebAssembly SIMD does not identify the host CPU, cache topology, browser JIT, sustained throughput, or memory bandwidth.".to_string(),
            simd_width_bits: 128,
            fp32_lanes: 4,
            fp16_lanes: 8,
            int8_lanes: 16,
            in_order: false,
            dot_product: false,
            sve2: false,
            xnnpack_kernel_family: "WASM SIMD 128-bit".to_string(),
            effective_memory_bandwidth_gbps: 10.0,
            effective_peak_gops: 180.0,
            compute_utilization_factor: 1.0,
            ridge_point_ops_per_byte: 18.0,
            memory_bound_intensity: 5.5,
            compute_bound_intensity: 18.0,
            // WASM SIMD: 16 INT8 vs 4 FP32 (4× hardware), ~2× effective (JIT overhead
            // and browser SIMD constraints limit INT8 packing gains).
            fp32_compute_factor: 2.0,
            int8_speedup_estimate: 1.25,
            channel_alignment_multiple: 8,
            weight_packing_bandwidth_gbps: 6.0,
            chain_break_overhead_us_low: 30.0,
            chain_break_overhead_us_high: 110.0,
        },
    ];
    for profile in &mut profiles {
        profile.profile_sha256 = target_profile_sha256(profile);
    }
    profiles
}

fn target_profile_sha256(profile: &TargetProfile) -> String {
    let basis = target_profile_hash_basis(profile);
    let digest = Sha256::digest(basis.as_bytes());
    hex_lower(&digest)
}

fn target_profile_f64_identity(value: f64) -> String {
    let canonical = if value == 0.0 { 0.0 } else { value };
    format!("{:016x}", canonical.to_bits())
}

fn target_profile_hash_basis(profile: &TargetProfile) -> String {
    let kernel_utilization = profile
        .compute_utilization_by_kernel_class
        .iter()
        .map(|(kernel_class, utilization)| {
            format!(
                "{}:{}",
                kernel_class,
                target_profile_f64_identity(*utilization)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    [
        "hash_schema=deepbom.target_profile_identity.v2".to_string(),
        format!("id={}", profile.id),
        format!("label={}", profile.label),
        format!(
            "derived_from={}",
            serde_json::to_string(&profile.derived_from).unwrap_or_else(|_| "null".to_string())
        ),
        format!("architecture={}", profile.architecture),
        format!("core_count_min={:?}", profile.core_count_min),
        format!("core_count_max={:?}", profile.core_count_max),
        format!(
            "performance_reference_core_count={:?}",
            profile.performance_reference_core_count
        ),
        format!("l1_data_bytes={}", profile.l1_data_bytes),
        format!("l2_bytes={}", profile.l2_bytes),
        format!("l2_capacity_scope={}", profile.l2_capacity_scope),
        format!("cache_assumption={}", profile.cache_assumption),
        format!("cache_source_url={}", profile.cache_source_url),
        format!(
            "hardware_spec={}",
            serde_json::to_string(&profile.hardware_spec).unwrap_or_else(|_| "null".to_string())
        ),
        format!(
            "performance_model_evidence_class={}",
            profile.performance_model_evidence_class
        ),
        format!(
            "performance_model_assumption={}",
            profile.performance_model_assumption
        ),
        format!("simd_width_bits={}", profile.simd_width_bits),
        format!("fp32_lanes={}", profile.fp32_lanes),
        format!("fp16_lanes={}", profile.fp16_lanes),
        format!("int8_lanes={}", profile.int8_lanes),
        format!("in_order={}", profile.in_order),
        format!("dot_product={}", profile.dot_product),
        format!("sve2={}", profile.sve2),
        format!("xnnpack_kernel_family={}", profile.xnnpack_kernel_family),
        format!(
            "effective_memory_bandwidth_gbps={}",
            target_profile_f64_identity(profile.effective_memory_bandwidth_gbps)
        ),
        format!(
            "effective_peak_gops={}",
            target_profile_f64_identity(profile.effective_peak_gops)
        ),
        format!(
            "compute_utilization_factor={}",
            target_profile_f64_identity(profile.compute_utilization_factor)
        ),
        format!("compute_utilization_by_kernel_class={kernel_utilization}"),
        format!(
            "ridge_point_ops_per_byte={}",
            target_profile_f64_identity(profile.ridge_point_ops_per_byte)
        ),
        format!(
            "memory_bound_intensity={}",
            target_profile_f64_identity(profile.memory_bound_intensity)
        ),
        format!(
            "compute_bound_intensity={}",
            target_profile_f64_identity(profile.compute_bound_intensity)
        ),
        format!(
            "fp32_compute_factor={}",
            target_profile_f64_identity(profile.fp32_compute_factor)
        ),
        format!(
            "int8_speedup_estimate={}",
            target_profile_f64_identity(profile.int8_speedup_estimate)
        ),
        format!(
            "channel_alignment_multiple={}",
            profile.channel_alignment_multiple
        ),
        format!(
            "weight_packing_bandwidth_gbps={}",
            target_profile_f64_identity(profile.weight_packing_bandwidth_gbps)
        ),
        format!(
            "chain_break_overhead_us_low={}",
            target_profile_f64_identity(profile.chain_break_overhead_us_low)
        ),
        format!(
            "chain_break_overhead_us_high={}",
            target_profile_f64_identity(profile.chain_break_overhead_us_high)
        ),
    ]
    .join("\n")
}

/// Accepts either a built-in profile id or a custom-profile specification:
///
/// ```json
/// {"base":"x86_avx2","id":"custom:my-laptop","label":"i7-14700HX 4T",
///  "evidence_class":"MEASURED",
///  "evidence_note":"compute_utilization_factor fitted to benchmark_model p50",
///  "overrides":{"compute_utilization_factor":0.05,"l2_bytes":2097152}}
/// ```
///
/// Routing every caller through here means analysis, the deployment frontier,
/// delegation repair, the research modules, and the native capture pipeline all
/// accept a custom target without further plumbing.
pub(super) fn target_profile(id: &str) -> Result<TargetProfile, String> {
    let trimmed = id.trim();
    if trimmed.starts_with('{') {
        return custom_target_profile(trimmed);
    }
    let mut profiles = all_target_profiles();
    match profiles.iter().position(|p| p.id == trimmed) {
        Some(i) => Ok(profiles.remove(i)),
        None => {
            let valid_ids: Vec<&str> = profiles.iter().map(|p| p.id.as_str()).collect();
            Err(format!(
                "Unknown target profile '{}'; valid IDs: {}",
                trimmed,
                valid_ids.join(", ")
            ))
        }
    }
}

const CUSTOM_TARGET_PREFIX: &str = "custom:";
const CUSTOM_TARGET_EVIDENCE_CLASSES: [&str; 3] = ["MEASURED", "VENDOR_DECLARED", "USER_DECLARED"];

fn custom_field<'a>(
    overrides: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a serde_json::Value> {
    overrides.get(key)
}

fn custom_f64(
    overrides: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    slot: &mut f64,
    min: f64,
    max: f64,
    changed: &mut Vec<String>,
) -> Result<(), String> {
    let Some(value) = custom_field(overrides, key) else {
        return Ok(());
    };
    let parsed = value
        .as_f64()
        .ok_or_else(|| format!("Custom target field {key} must be a number"))?;
    if !parsed.is_finite() || parsed < min || parsed > max {
        return Err(format!(
            "Custom target field {key} must be a finite number in [{min}, {max}]; received {parsed}"
        ));
    }
    *slot = parsed;
    changed.push(key.to_string());
    Ok(())
}

fn custom_usize(
    overrides: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    slot: &mut usize,
    min: u64,
    max: u64,
    changed: &mut Vec<String>,
) -> Result<(), String> {
    let Some(value) = custom_field(overrides, key) else {
        return Ok(());
    };
    let parsed = value
        .as_u64()
        .ok_or_else(|| format!("Custom target field {key} must be a non-negative integer"))?;
    if parsed < min || parsed > max {
        return Err(format!(
            "Custom target field {key} must be in [{min}, {max}]; received {parsed}"
        ));
    }
    *slot = parsed as usize;
    changed.push(key.to_string());
    Ok(())
}

fn custom_option_usize(
    overrides: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    slot: &mut Option<usize>,
    min: u64,
    max: u64,
    changed: &mut Vec<String>,
) -> Result<(), String> {
    let Some(value) = custom_field(overrides, key) else {
        return Ok(());
    };
    let parsed = value
        .as_u64()
        .ok_or_else(|| format!("Custom target field {key} must be a positive integer"))?;
    if parsed < min || parsed > max {
        return Err(format!(
            "Custom target field {key} must be in [{min}, {max}]; received {parsed}"
        ));
    }
    *slot = Some(parsed as usize);
    changed.push(key.to_string());
    Ok(())
}

fn custom_bool(
    overrides: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    slot: &mut bool,
    changed: &mut Vec<String>,
) -> Result<(), String> {
    let Some(value) = custom_field(overrides, key) else {
        return Ok(());
    };
    let parsed = value
        .as_bool()
        .ok_or_else(|| format!("Custom target field {key} must be a boolean"))?;
    *slot = parsed;
    changed.push(key.to_string());
    Ok(())
}

fn custom_string(
    overrides: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    slot: &mut String,
    max_len: usize,
    changed: &mut Vec<String>,
) -> Result<(), String> {
    let Some(value) = custom_field(overrides, key) else {
        return Ok(());
    };
    let parsed = value
        .as_str()
        .ok_or_else(|| format!("Custom target field {key} must be a string"))?;
    if parsed.len() > max_len {
        return Err(format!(
            "Custom target field {key} exceeds {max_len} characters"
        ));
    }
    *slot = parsed.to_string();
    changed.push(key.to_string());
    Ok(())
}

pub(super) fn custom_target_profile(spec: &str) -> Result<TargetProfile, String> {
    if spec.len() > 16_384 {
        return Err("Custom target specification exceeds 16384 bytes".to_string());
    }
    let root: serde_json::Value = serde_json::from_str(spec)
        .map_err(|error| format!("Custom target specification is not valid JSON: {error}"))?;
    let root = root
        .as_object()
        .ok_or_else(|| "Custom target specification must be a JSON object".to_string())?;

    let base_id = root
        .get("base")
        .and_then(|value| value.as_str())
        .ok_or_else(|| "Custom target specification requires a base profile id".to_string())?;
    if base_id.trim_start().starts_with('{') {
        return Err("Custom target base must be a built-in profile id".to_string());
    }
    let mut profile = target_profile(base_id)?;
    let base_sha256 = profile.profile_sha256.clone();

    let id = root
        .get("id")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .trim()
        .to_string();
    if !id.starts_with(CUSTOM_TARGET_PREFIX) || id.len() <= CUSTOM_TARGET_PREFIX.len() {
        return Err(format!(
            "Custom target id must start with '{CUSTOM_TARGET_PREFIX}' so it can never collide with a built-in profile"
        ));
    }
    if id.len() > 128 || !id.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
        return Err("Custom target id must be at most 128 printable ASCII characters".to_string());
    }
    let label = root
        .get("label")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .trim()
        .to_string();
    if label.is_empty() || label.len() > 128 {
        return Err("Custom target requires a label of 1-128 characters".to_string());
    }

    let empty = serde_json::Map::new();
    let overrides = match root.get("overrides") {
        Some(value) => value
            .as_object()
            .ok_or_else(|| "Custom target overrides must be a JSON object".to_string())?,
        None => &empty,
    };

    const TUNABLE_FIELDS: [&str; 25] = [
        "core_count_min",
        "core_count_max",
        "performance_reference_core_count",
        "l1_data_bytes",
        "l2_bytes",
        "l2_capacity_scope",
        "architecture",
        "cache_assumption",
        "cache_source_url",
        "xnnpack_kernel_family",
        "simd_width_bits",
        "fp32_lanes",
        "fp16_lanes",
        "int8_lanes",
        "channel_alignment_multiple",
        "in_order",
        "dot_product",
        "sve2",
        "effective_memory_bandwidth_gbps",
        "effective_peak_gops",
        "compute_utilization_factor",
        "fp32_compute_factor",
        "int8_speedup_estimate",
        "weight_packing_bandwidth_gbps",
        "chain_break_overhead_us_low",
    ];
    for key in overrides.keys() {
        if !TUNABLE_FIELDS.contains(&key.as_str())
            && key != "chain_break_overhead_us_high"
            && key != "compute_utilization_by_kernel_class"
        {
            return Err(format!(
                "Custom target override '{key}' is not a tunable field; tunable fields: {}, chain_break_overhead_us_high, compute_utilization_by_kernel_class",
                TUNABLE_FIELDS.join(", ")
            ));
        }
    }

    let mut changed: Vec<String> = Vec::new();
    custom_option_usize(
        overrides,
        "core_count_min",
        &mut profile.core_count_min,
        1,
        1024,
        &mut changed,
    )?;
    custom_option_usize(
        overrides,
        "core_count_max",
        &mut profile.core_count_max,
        1,
        1024,
        &mut changed,
    )?;
    custom_option_usize(
        overrides,
        "performance_reference_core_count",
        &mut profile.performance_reference_core_count,
        1,
        1024,
        &mut changed,
    )?;
    if let (Some(minimum), Some(maximum)) = (profile.core_count_min, profile.core_count_max) {
        if maximum < minimum {
            return Err(
                "Custom target core_count_max must not be below core_count_min".to_string(),
            );
        }
        if let Some(reference) = profile.performance_reference_core_count {
            if reference != minimum && reference != maximum {
                return Err("Custom target performance_reference_core_count must match one declared system core-count variant".to_string());
            }
        }
    }
    custom_usize(
        overrides,
        "l1_data_bytes",
        &mut profile.l1_data_bytes,
        1024,
        1 << 24,
        &mut changed,
    )?;
    custom_usize(
        overrides,
        "l2_bytes",
        &mut profile.l2_bytes,
        4096,
        1 << 30,
        &mut changed,
    )?;
    custom_usize(
        overrides,
        "simd_width_bits",
        &mut profile.simd_width_bits,
        32,
        2048,
        &mut changed,
    )?;
    custom_usize(
        overrides,
        "fp32_lanes",
        &mut profile.fp32_lanes,
        1,
        256,
        &mut changed,
    )?;
    custom_usize(
        overrides,
        "fp16_lanes",
        &mut profile.fp16_lanes,
        1,
        512,
        &mut changed,
    )?;
    custom_usize(
        overrides,
        "int8_lanes",
        &mut profile.int8_lanes,
        1,
        1024,
        &mut changed,
    )?;
    custom_usize(
        overrides,
        "channel_alignment_multiple",
        &mut profile.channel_alignment_multiple,
        1,
        256,
        &mut changed,
    )?;
    custom_bool(overrides, "in_order", &mut profile.in_order, &mut changed)?;
    custom_bool(
        overrides,
        "dot_product",
        &mut profile.dot_product,
        &mut changed,
    )?;
    custom_bool(overrides, "sve2", &mut profile.sve2, &mut changed)?;
    custom_string(
        overrides,
        "architecture",
        &mut profile.architecture,
        128,
        &mut changed,
    )?;
    custom_string(
        overrides,
        "l2_capacity_scope",
        &mut profile.l2_capacity_scope,
        128,
        &mut changed,
    )?;
    custom_string(
        overrides,
        "cache_assumption",
        &mut profile.cache_assumption,
        1024,
        &mut changed,
    )?;
    custom_string(
        overrides,
        "cache_source_url",
        &mut profile.cache_source_url,
        512,
        &mut changed,
    )?;
    custom_string(
        overrides,
        "xnnpack_kernel_family",
        &mut profile.xnnpack_kernel_family,
        64,
        &mut changed,
    )?;
    custom_f64(
        overrides,
        "effective_memory_bandwidth_gbps",
        &mut profile.effective_memory_bandwidth_gbps,
        0.25,
        10_000.0,
        &mut changed,
    )?;
    custom_f64(
        overrides,
        "effective_peak_gops",
        &mut profile.effective_peak_gops,
        0.1,
        1_000_000.0,
        &mut changed,
    )?;
    custom_f64(
        overrides,
        "compute_utilization_factor",
        &mut profile.compute_utilization_factor,
        0.01,
        1.0,
        &mut changed,
    )?;
    custom_f64(
        overrides,
        "fp32_compute_factor",
        &mut profile.fp32_compute_factor,
        0.1,
        64.0,
        &mut changed,
    )?;
    custom_f64(
        overrides,
        "int8_speedup_estimate",
        &mut profile.int8_speedup_estimate,
        0.1,
        64.0,
        &mut changed,
    )?;
    custom_f64(
        overrides,
        "weight_packing_bandwidth_gbps",
        &mut profile.weight_packing_bandwidth_gbps,
        0.1,
        10_000.0,
        &mut changed,
    )?;
    custom_f64(
        overrides,
        "chain_break_overhead_us_low",
        &mut profile.chain_break_overhead_us_low,
        0.0,
        100_000.0,
        &mut changed,
    )?;
    custom_f64(
        overrides,
        "chain_break_overhead_us_high",
        &mut profile.chain_break_overhead_us_high,
        0.0,
        100_000.0,
        &mut changed,
    )?;
    if profile.chain_break_overhead_us_high < profile.chain_break_overhead_us_low {
        return Err("Custom target chain_break_overhead_us_high must not be below chain_break_overhead_us_low".to_string());
    }

    if let Some(value) = overrides.get("compute_utilization_by_kernel_class") {
        let map = value.as_object().ok_or_else(|| {
            "Custom target compute_utilization_by_kernel_class must be a JSON object".to_string()
        })?;
        if map.len() > 64 {
            return Err(
                "Custom target compute_utilization_by_kernel_class accepts at most 64 families"
                    .to_string(),
            );
        }
        let mut resolved = BTreeMap::new();
        for (family, entry) in map {
            if family.is_empty()
                || family.len() > 64
                || !family
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            {
                return Err(format!(
                    "Custom target kernel family '{family}' must be lower_snake_case ASCII of at most 64 characters"
                ));
            }
            let parsed = entry.as_f64().ok_or_else(|| {
                format!("Custom target utilization for '{family}' must be a number")
            })?;
            if !parsed.is_finite() || !(0.01..=1.0).contains(&parsed) {
                return Err(format!(
                    "Custom target utilization for '{family}' must be in [0.01, 1]; received {parsed}"
                ));
            }
            resolved.insert(family.clone(), parsed);
        }
        profile.compute_utilization_entries = resolved
            .iter()
            .map(|(kernel_class, utilization)| KernelUtilizationEntry {
                kernel_class: kernel_class.clone(),
                utilization: *utilization,
            })
            .collect();
        profile.compute_utilization_by_kernel_class = resolved;
        changed.push("compute_utilization_by_kernel_class".to_string());
    }

    changed.sort();
    changed.dedup();

    // Derived roofline thresholds must follow the numbers they are derived from,
    // otherwise a retuned profile keeps the base profile's bound classification.
    if changed.iter().any(|key| {
        key == "effective_peak_gops"
            || key == "compute_utilization_factor"
            || key == "effective_memory_bandwidth_gbps"
    }) {
        let effective_gops =
            profile.effective_peak_gops * profile.compute_utilization_factor.clamp(0.01, 1.0);
        let ridge = effective_gops / profile.effective_memory_bandwidth_gbps.max(0.25);
        profile.ridge_point_ops_per_byte = ridge;
        profile.compute_bound_intensity = ridge;
        profile.memory_bound_intensity = ridge / 3.4;
    }

    let evidence_class = root
        .get("evidence_class")
        .and_then(|value| value.as_str())
        .unwrap_or("USER_DECLARED")
        .to_string();
    if !CUSTOM_TARGET_EVIDENCE_CLASSES.contains(&evidence_class.as_str()) {
        return Err(format!(
            "Custom target evidence_class must be one of: {}",
            CUSTOM_TARGET_EVIDENCE_CLASSES.join(", ")
        ));
    }
    let evidence_note = root
        .get("evidence_note")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if evidence_note.len() > 1024 {
        return Err("Custom target evidence_note exceeds 1024 characters".to_string());
    }
    if evidence_class == "MEASURED" && evidence_note.is_empty() {
        return Err(
            "Custom target evidence_class MEASURED requires an evidence_note naming the measurement"
                .to_string(),
        );
    }

    // The base profile's assumption text describes the base profile's numbers.
    // Retuned numbers must not inherit it, or the artifact would carry a claim
    // that is no longer true of the profile actually used.
    profile.performance_model_assumption = format!(
        "Custom profile derived from built-in '{}' ({}). Retuned fields: {}. Evidence class {}{}. \
Fields not listed here retain the base profile's assumptions.",
        base_id,
        base_sha256,
        if changed.is_empty() {
            "none".to_string()
        } else {
            changed.join(", ")
        },
        evidence_class,
        if evidence_note.is_empty() {
            String::new()
        } else {
            format!(" — {evidence_note}")
        },
    );
    profile.performance_model_evidence_class = evidence_class;
    profile.id = id;
    profile.label = label;
    profile.derived_from = Some(TargetProfileDerivation {
        base_profile_id: base_id.to_string(),
        base_profile_sha256: base_sha256,
        overridden_fields: changed,
        evidence_note,
    });
    profile.profile_sha256 = target_profile_sha256(&profile);
    Ok(profile)
}
