use num_traits::Zero;

pub fn drop_in_background_thread<T>(data: T)
where
    T: Send + 'static,
{
    // h/t https://abrams.cc/rust-dropping-things-in-another-thread
    rayon::spawn(move || drop(data));
}

pub fn unsafe_allocate_zero_vec<T: Sized + Zero>(size: usize) -> Vec<T> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value

    #[cfg(test)]
    {
        // Check for safety of 0 allocation
        unsafe {
            let value = &T::zero();
            let ptr = value as *const T as *const u8;
            let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<T>());
            assert!(bytes.iter().all(|&byte| byte == 0));
        }
    }

    // Bulk allocate zeros, unsafely
    let result: Vec<T>;
    unsafe {
        let layout = std::alloc::Layout::array::<T>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut T;

        if ptr.is_null() {
            panic!("Zero vec allocation failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}

/// Allocates a zero-initialized Vec<T> with a requested alignment (in bytes).
/// Caller must ensure `align` is a power of two and >= align_of::<T>().
/// Falls back to align_of::<T>() if a smaller alignment is requested.
pub fn unsafe_allocate_zero_vec_aligned<T: Sized + Zero>(size: usize, align: usize) -> Vec<T> {
    let required_align = core::mem::align_of::<T>();
    let use_align = if align >= required_align { align } else { required_align };
    let bytes = size.checked_mul(core::mem::size_of::<T>()).expect("size overflow");

    let result: Vec<T>;
    unsafe {
        let layout = std::alloc::Layout::from_size_align(bytes, use_align).expect("invalid layout");
        let ptr = std::alloc::alloc_zeroed(layout) as *mut T;
        if ptr.is_null() { panic!("Zero vec aligned allocation failed"); }
        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}
