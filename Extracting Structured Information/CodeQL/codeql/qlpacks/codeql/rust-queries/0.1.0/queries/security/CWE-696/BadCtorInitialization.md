# Bad 'ctor' initialization
Calling functions and methods in the Rust `std` library from a `#[ctor]` or `#[dtor]` function is not safe. This is because the `std` library only guarantees stability and portability between the beginning and the end of `main`, whereas `#[ctor]` functions are called before `main`, and `#[dtor]` functions are called after it.


## Recommendation
Do not call any part of the `std` library from a `#[ctor]` or `#[dtor]` function. Instead either:

* Move the code to a different location, such as inside your program's `main` function.
* Rewrite the code using an alternative library.

## Example
In the following example, a `#[ctor]` function uses the `println!` macro which calls `std` library functions. This may cause unexpected behavior at runtime.


```rust

#[ctor::ctor]
fn bad_example() {
    println!("Hello, world!"); // BAD: the println! macro calls std library functions
}

```
The issue can be fixed by replacing `println!` with something that does not rely on the `std` library. In the fixed code below, we used the `libc_println!` macro from the `libc-print` library:


```rust

#[ctor::ctor]
fn good_example() {
    libc_print::libc_println!("Hello, world!"); // GOOD: libc-print does not use the std library
}

```

## References
* GitHub: [rust-ctor - Warnings](https://github.com/mmastrac/rust-ctor?tab=readme-ov-file#warnings).
* Rust Programming Language: [Crate std - Use before and after main()](https://doc.rust-lang.org/std/#use-before-and-after-main).
* Common Weakness Enumeration: [CWE-696](https://cwe.mitre.org/data/definitions/696.html).
* Common Weakness Enumeration: [CWE-665](https://cwe.mitre.org/data/definitions/665.html).
