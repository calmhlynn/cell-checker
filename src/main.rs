use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

fn generic_fill_and_check(arr: &mut [u16]) -> bool {
    let mut flag = true;

    let _max = arr.len();

    for i in 0.._max {
        arr[i] = 1;
    }

    for i in 0.._max {
        if arr[i] != 1 {
            flag = false;
        }
    }

    flag
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_fill_and_check(arr: &mut [u16]) -> bool {
    let len = arr.len();
    let ptr = arr.as_mut_ptr();
    let step = 8;
    let num_chunks = len / step;

    unsafe {
        let ones = vdupq_n_u16(1);

        for i in 0..num_chunks {
            let chunk_ptr = ptr.add(i * step);
            vst1q_u16(chunk_ptr, ones);

            let data = vld1q_u16(chunk_ptr);

            let comparison = vceqq_u16(data, ones);

            if vminvq_u16(comparison) == 0 {
                return false;
            }
        }

        for i in (num_chunks * step)..len {
            let current_ptr = ptr.add(i);
            *current_ptr = 1;
            if *current_ptr != 1 {
                return false;
            }
        }
    }

    true
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_fill_only(arr: &mut [u16]) {
    let len = arr.len();
    let ptr = arr.as_mut_ptr();
    let step = 8;
    let num_chunks = len / step;

    unsafe {
        let ones = vdupq_n_u16(1);

        for i in 0..num_chunks {
            let chunk_ptr = ptr.add(i * step);
            vst1q_u16(chunk_ptr, ones);
        }

        for i in (num_chunks * step)..len {
            *ptr.add(i) = 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_check_only(arr: &[u16]) -> bool {
    let len = arr.len();
    let ptr = arr.as_ptr();
    let step = 8;
    let num_chunks = len / step;

    unsafe {
        let ones = vdupq_n_u16(1);

        for i in 0..num_chunks {
            let chunk_ptr = ptr.add(i * step);
            let data = vld1q_u16(chunk_ptr);
            let comparison = vceqq_u16(data, ones);

            if vminvq_u16(comparison) == 0 {
                return false;
            }
        }

        for i in (num_chunks * step)..len {
            if *ptr.add(i) != 1 {
                return false;
            }
        }
    }

    true
}

#[cfg(target_arch = "aarch64")]
unsafe fn simd_fill_and_check_optimized(arr: &mut [u16]) -> bool {
    unsafe {
        simd_fill_only(arr);
        simd_check_only(arr)
    }
}

fn main() {
    let n = 4096 * 10;
    let array_size = n * n;
    println!("Array size: {}x{} (total {} elements)\n", n, n, array_size);

    let mut arr = vec![0u16; array_size];

    // generic func
    let start_generic = Instant::now();
    let result_generic = generic_fill_and_check(&mut arr);
    println!(
        "Generic function (2-pass) result: {}, time: {:?}",
        result_generic,
        start_generic.elapsed()
    );

    #[cfg(target_arch = "aarch64")]
    {
        // --- Fused SIMD ---
        let start_simd_fused = Instant::now();
        let result_simd_fused = unsafe { simd_fill_and_check(&mut arr) };
        println!(
            "Existing SIMD (Fused) result: {}, time: {:?}",
            result_simd_fused,
            start_simd_fused.elapsed()
        );

        // --- Separated SIMD  ---
        let start_simd_optimized = Instant::now();
        let result_simd_optimized = unsafe { simd_fill_and_check_optimized(&mut arr) };
        println!(
            "Optimized SIMD (Separated) result: {}, time: {:?}",
            result_simd_optimized,
            start_simd_optimized.elapsed()
        );
    }
}
