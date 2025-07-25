use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

fn generic_fill_and_check(arr: &mut [u16]) -> bool {
    for i in 0..arr.len() {
        arr[i] = 1;
    }

    for i in 0..arr.len() {
        if arr[i] != 1 {
            return false;
        }
    }

    true
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_fill_and_check(arr: &mut [u16]) -> bool {
    let len = arr.len();
    let ptr = arr.as_mut_ptr();
    let step = 8;
    let num_chunks = len / step;

    let ones = vdupq_n_u16(1);

    for i in 0..num_chunks {
        let chunk_ptr = ptr.add(i * step);
        vst1q_u16(chunk_ptr, ones);
    }

    for i in (num_chunks * step)..len {
        *ptr.add(i) = 1;
    }

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

    true
}

fn main() {
    let n = 512;
    let array_size = n * n;

    println!("--- Array Fill and Verification Performance Measurement ---");
    println!("Array size: {}x{} (total {} elements)\n", n, n, array_size);

    let mut arr_generic = vec![0u16; array_size];
    let start_generic = Instant::now();
    let result_generic = generic_fill_and_check(&mut arr_generic);
    let duration_generic = start_generic.elapsed();
    println!(
        "Generic function result: {}, time: {:?}",
        result_generic, duration_generic
    );

    #[cfg(target_arch = "aarch64")]
    {
        let mut arr_simd = vec![0u16; array_size];
        let start_simd = Instant::now();
        let result_simd = unsafe { simd_fill_and_check(&mut arr_simd) };
        let duration_simd = start_simd.elapsed();
        println!("SIMD function result: {}, time: {:?}", result_simd, duration_simd);
    }

    println!("------------------------------------");
}
