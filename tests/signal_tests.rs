use approx::assert_relative_eq;
use dsp4rust::signal::Signal;

#[test]
fn test_signal_creation() {
    // 测试 from_vec
    let vec_signal = Signal::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(vec_signal.len(), 5);
    assert_eq!(vec_signal.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    // 测试 from_elem
    let elem_signal = Signal::from_elem(3.14, 3);
    assert_eq!(elem_signal.len(), 3);
    assert_eq!(elem_signal.to_vec(), vec![3.14, 3.14, 3.14]);

    // 测试 from_len_fn
    let fn_signal = Signal::from_len_fn(5, |i| (i as f64) * 2.0);
    assert_eq!(fn_signal.len(), 5);
    assert_eq!(fn_signal.to_vec(), vec![0.0, 2.0, 4.0, 6.0, 8.0]);

    // 测试 from_iter
    let iter_signal: Signal = (0..5).map(|i| i as f64 * 2.0).collect();
    assert_eq!(iter_signal.len(), 5);
    assert_eq!(iter_signal.to_vec(), vec![0.0, 2.0, 4.0, 6.0, 8.0]);

    // 测试 zeros, ones, linspace, arrange
    let zeros = Signal::zeros(3);
    assert_eq!(zeros.len(), 3);
    assert_eq!(zeros.to_vec(), vec![0.0, 0.0, 0.0]);

    let ones = Signal::ones(3);
    assert_eq!(ones.len(), 3);
    assert_eq!(ones.to_vec(), vec![1.0, 1.0, 1.0]);

    let linspace = Signal::linspace(0.0, 1.0, 5);
    assert_eq!(linspace.len(), 5);
    assert_relative_eq!(linspace.to_vec()[0], 0.0);
    assert_relative_eq!(linspace.to_vec()[1], 0.25);
    assert_relative_eq!(linspace.to_vec()[2], 0.5);
    assert_relative_eq!(linspace.to_vec()[3], 0.75);
    assert_relative_eq!(linspace.to_vec()[4], 1.0);

    let arrange = Signal::arrange(0.0, 1.0, 0.25);
    assert_eq!(arrange.len(), 4);
    assert_eq!(arrange.to_vec(), vec![0.0, 0.25, 0.5, 0.75]);
}
