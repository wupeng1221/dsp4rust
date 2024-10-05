use std::f64::consts::TAU;
use ndarray_rand::rand_distr::{Distribution, Normal};
use ndarray_rand::rand;
use crate::signal::Signal;

/// Signal generator for various waveforms.
///
/// This struct provides methods to generate common signal types such as sine, square, triangle,
/// and sawtooth waves, as well as noise and custom waveforms.
///
/// 信号生成器，用于生成各种波形。
///
/// 该结构体提供生成常见信号类型的方法，如正弦波、方波、三角波和锯齿波，以及噪声和自定义波形。
///
/// # Attributes
///
/// * `sample_rate` - The number of samples per second (Hz) / 每秒采样数（赫兹）
/// * `start_time` - The start time of the signal (seconds) / 信号的起始时间（秒）
/// * `stop_time` - The stop time of the signal (seconds) / 信号的结束时间（秒）
///
/// # Construction
///
/// Use the `new()` method to create a default instance, then chain the builder methods
/// to set the desired attributes:
///
/// 使用 `new()` 方法创建默认实例，然后链式调用构建器方法设置所需的属性：
///
/// ```
/// use dsp4rust::generator::Generator;
///
/// let generator = Generator::new()
///     .sample_rate(44100.0)
///     .start_time(0.0)
///     .stop_time(1.0)
///     .build();
/// ```
///
/// # Modifying Attributes
///
/// The attributes can only be modified during the construction phase using the builder pattern.
/// Once built, the Generator instance is immutable.
///
/// 属性只能在构造阶段使用构建器模式进行修改。一旦构建完成，Generator 实例就是不可变的。
///
/// # Examples
///
/// Generating a sine wave:
///
/// 生成正弦波：
///
/// ```
/// use dsp4rust::generator::Generator;
///
/// let generator = Generator::new()
///     .sample_rate(44100.0)
///     .start_time(0.0)
///     .stop_time(1.0)
///     .build();
///
/// let sine_wave = generator.sin_unit(440.0, 0.0);
/// ```
#[derive(Default)]
pub struct Generator {
    sample_rate: f64,
    start_time: f64,
    stop_time: f64,
}

impl Generator {
    /// Creates a new Generator instance with default values.
    ///
    /// 创建一个具有默认值的新 Generator 实例。
    ///
    /// # Examples
    ///
    /// ```
    /// use dsp4rust::generator::Generator;
    ///
    /// let generator = Generator::new();
    /// ```
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Default::default()
    }

    /// Sets the sample rate.
    ///
    /// 设置采样率。
    ///
    /// # Examples
    ///
    /// ```
    /// use dsp4rust::generator::Generator;
    ///
    /// let generator = Generator::new().sample_rate(44100.0);
    /// ```
    #[must_use]
    pub fn sample_rate(mut self, sample_rate: f64) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Sets the start time.
    ///
    /// 设置起始时间。
    ///
    /// # Examples
    ///
    /// ```
    /// use dsp4rust::generator::Generator;
    ///
    /// let generator = Generator::new().start_time(0.0);
    /// ```
    #[must_use]
    pub fn start_time(mut self, start_time: f64) -> Self {
        self.start_time = start_time;
        self
    }

    /// Sets the stop time.
    ///
    /// 设置结束时间。
    ///
    /// # Examples
    ///
    /// ```
    /// use dsp4rust::generator::Generator;
    ///
    /// let generator = Generator::new().stop_time(1.0);
    /// ```
    #[must_use]
    pub fn stop_time(mut self, stop_time: f64) -> Self {
        self.stop_time = stop_time;
        self
    }

    /// Builds the Generator.
    ///
    /// 构建 Generator。
    ///
    /// # Examples
    ///
    /// ```
    /// use dsp4rust::generator::Generator;
    ///
    /// let generator = Generator::new()
    ///     .sample_rate(44100.0)
    ///     .start_time(0.0)
    ///     .stop_time(1.0)
    ///     .build();
    /// ```
    pub fn build(self) -> Self {
        self
    }

    /// Generates a unit sine wave signal.
    ///
    /// 生成单位正弦波信号。
    ///
    /// # Examples
    ///
    /// ```
    /// # use dsp4rust::generator::Generator;
    /// let generator = Generator::new().sample_rate(44100.0).start_time(0.0).stop_time(1.0).build();
    /// let sine = generator.sin_unit(440.0, 0.0);
    /// ```
    pub fn sin_unit(&self, freq: f64, phase: f64) -> Signal {
        let samples = ((self.stop_time - self.start_time) * self.sample_rate) as usize;
        let mut data = Vec::with_capacity(samples);

        let mut t = self.start_time;
        let sample_gap = 1.0 / self.sample_rate;

        for _ in 0..samples {
            data.push((TAU * freq * t + phase).sin());
            t += sample_gap;
        }

        Signal::from_vec(data)
    }

    /// Generates a unit pulse wave signal.
    ///
    /// 生成单位脉冲波信号。
    ///
    /// # Parameters
    ///
    /// * `phase` - Phase in radians. Range: [0, 2π) / 相位（弧度）。范围：[0, 2π)
    /// * `duty_cycle` - Duty cycle. Range: [0.0, 1.0] / 占空比。范围：[0.0, 1.0]
    ///
    /// # Examples
    ///
    /// ```
    /// # use dsp4rust::generator::Generator;
    /// let generator = Generator::new().sample_rate(44100.0).start_time(0.0).stop_time(1.0).build();
    /// let pulse = generator.pulse_unit(440.0, 0.0, 0.5);
    /// ```
    pub fn pulse_unit(&self, freq: f64, phase: f64, duty_cycle: f64) -> Signal {
        let samples = ((self.stop_time - self.start_time) * self.sample_rate) as usize;
        let mut data = Vec::with_capacity(samples);

        let period_samples = (self.sample_rate / freq) as usize;
        let high_samples = (duty_cycle * period_samples as f64) as usize;
        let phase_offset = ((phase / TAU) * period_samples as f64).round() as usize;

        for i in 0..samples {
            let t_in_period = (i + phase_offset) % period_samples;
            if t_in_period < high_samples {
                data.push(1.0)
            } else {
                data.push(-1.0)
            }
        }

        Signal::from_vec(data)
    }

    /// Generates a unit square wave signal.
    ///
    /// 生成单位方波信号。
    ///
    /// # Examples
    ///
    /// ```
    /// # use dsp4rust::generator::Generator;
    /// let generator = Generator::new().sample_rate(44100.0).start_time(0.0).stop_time(1.0).build();
    /// let square = generator.square_unit(440.0, 0.0);
    /// ```
    pub fn square_unit(&self, freq: f64, phase: f64) -> Signal {
        self.pulse_unit(freq, phase, 0.5)
    }

    /// Generates a unit triangle wave signal.
    ///
    /// 生成单位三角波信号。
    ///
    /// # Examples
    ///
    /// ```
    /// # use dsp4rust::generator::Generator;
    /// let generator = Generator::new().sample_rate(44100.0).start_time(0.0).stop_time(1.0).build();
    /// let triangle = generator.triangle_unit(440.0, 0.0);
    /// ```
    pub fn triangle_unit(&self, freq: f64, phase: f64) -> Signal {
        let samples = ((self.stop_time - self.start_time) * self.sample_rate) as usize;
        let mut data = Vec::with_capacity(samples);

        let period_samples = (self.sample_rate / freq) as usize;
        let phase_offset = ((phase / TAU) * period_samples as f64).round() as usize;

        for i in 0..samples {
            let t_in_period = (i + phase_offset) % period_samples;
            let normalized_t = t_in_period as f64 / period_samples as f64;

            let value = if normalized_t < 0.5 {
                4.0 * normalized_t - 1.0
            } else {
                1.0 - 4.0 * (normalized_t - 0.5)
            };

            data.push(value);
        }

        Signal::from_vec(data)
    }

    /// Generates a unit sawtooth wave signal.
    ///
    /// 生成单位锯齿波信号。
    ///
    /// # Examples
    ///
    /// ```
    /// # use dsp4rust::generator::Generator;
    /// let generator = Generator::new().sample_rate(44100.0).start_time(0.0).stop_time(1.0).build();
    /// let sawtooth = generator.sawtooth_unit(440.0, 0.0);
    /// ```
    pub fn sawtooth_unit(&self, freq: f64, phase: f64) -> Signal {
        let samples = ((self.stop_time - self.start_time) * self.sample_rate) as usize;
        let mut data = Vec::with_capacity(samples);
        let period_samples = (self.sample_rate / freq) as usize;
        let phase_offset = ((phase / TAU) * period_samples as f64).round() as usize;

        for i in 0..samples {
            let t_in_period = (i + phase_offset) % period_samples;

            let normalized_t = t_in_period as f64 / period_samples as f64;

            let value = 2.0 * (normalized_t - 0.5);

            data.push(value);
        }

        Signal::from_vec(data)
    }

    /// Generates a unit step signal.
    ///
    /// 生成单位阶跃信号。
    ///
    /// # Examples
    ///
    /// ```
    /// # use dsp4rust::generator::Generator;
    /// let generator = Generator::new().sample_rate(44100.0).start_time(0.0).stop_time(1.0).build();
    /// let step = generator.step_unit(0.5);
    /// ```
    pub fn step_unit(&self, step_time: f64) -> Signal {
        let samples = ((self.stop_time - self.start_time) * self.sample_rate) as usize;
        let mut data = Vec::with_capacity(samples);

        for i in 0..samples {
            let t = self.start_time + (i as f64 / self.sample_rate);

            if t < step_time {
                data.push(0.0);
            } else {
                data.push(1.0);
            }
        }

        Signal::from_vec(data)
    }

    /// Generates a Gaussian pulse signal.
    ///
    /// 生成高斯脉冲信号。
    ///
    /// # Examples
    ///
    /// ```
    /// # use dsp4rust::generator::Generator;
    /// let generator = Generator::new().sample_rate(44100.0).start_time(0.0).stop_time(1.0).build();
    /// let gaussian_pulse = generator.gaussian_pulse(0.5, 0.1);
    /// ```
    pub fn gaussian_pulse(&self, center_time: f64, sigma: f64) -> Signal {
        let samples = ((self.stop_time - self.start_time) * self.sample_rate) as usize;
        let mut data = Vec::with_capacity(samples);

        let mut t = self.start_time;
        let sample_gap = 1.0 / self.sample_rate;

        for _ in 0..samples {
            let value = (-((t - center_time).powi(2)) / (2.0 * sigma.powi(2))).exp();
            data.push(value);
            t += sample_gap;
        }

        Signal::from_vec(data)
    }

    /// Generates a unit Gaussian pulse signal.
    ///
    /// 生成单位高斯脉冲信号。
    ///
    /// # Examples
    ///
    /// ```
    /// # use dsp4rust::generator::Generator;
    /// let generator = Generator::new().sample_rate(44100.0).start_time(0.0).stop_time(1.0).build();
    /// let gaussian_unit = generator.gaussian_unit();
    /// ```
    pub fn gaussian_unit(&self) -> Signal {
        self.gaussian_pulse(0.0, 1.0)
    }

    /// Generates Gaussian white noise.
    ///
    /// 生成高斯白噪声。
    ///
    /// # Examples
    ///
    /// ```
    /// # use dsp4rust::generator::Generator;
    /// let generator = Generator::new().sample_rate(44100.0).start_time(0.0).stop_time(1.0).build();
    /// let noise = generator.gaussian_white_noise(0.0, 1.0);
    /// ```
    pub fn gaussian_white_noise(&self, mean: f64, std_dev: f64) -> Signal {
        let samples = ((self.stop_time - self.start_time) * self.sample_rate) as usize;

        // 创建一个服从正态分布的随机数组
        let noise_vec: Vec<f64> = (0..samples)
            .map(|_| Normal::new(mean, std_dev).unwrap().sample(&mut rand::thread_rng()))
            .collect();

        Signal::from_vec(noise_vec)
    }

    /// Generates an exponential signal.
    ///
    /// 生成指数信号。
    ///
    /// # Examples
    ///
    /// ```
    /// # use dsp4rust::generator::Generator;
    /// let generator = Generator::new().sample_rate(44100.0).start_time(0.0).stop_time(1.0).build();
    /// let exp_signal = generator.exponential_signal(2.0);
    /// ```
    pub fn exponential_signal(&self, alpha: f64) -> Signal {
        let samples = ((self.stop_time - self.start_time) * self.sample_rate) as usize;
        let mut data = Vec::with_capacity(samples);

        let mut t = self.start_time;
        let sample_gap = 1.0 / self.sample_rate;

        for _ in 0..samples {
            data.push((alpha * t).exp());
            t += sample_gap;
        }

        Signal::from_vec(data)
    }

    /// Generates a custom waveform based on the provided function.
    ///
    /// 根据提供的函数生成自定义波形。
    ///
    /// # Examples
    ///
    /// ```
    /// # use dsp4rust::generator::Generator;
    /// let generator = Generator::new().sample_rate(44100.0).start_time(0.0).stop_time(1.0).build();
    /// let custom_wave = generator.fn_wave(|t| t.sin() + 0.5 * (2.0 * t).sin());
    /// ```
    pub fn fn_wave<T>(&self, f: T) -> Signal where T: Fn(f64) -> f64 {
        let samples = ((self.stop_time - self.start_time) * self.sample_rate) as usize;
        let mut data = Vec::with_capacity(samples);

        let mut t = 0.0;
        let sample_gap = 1.0 / self.sample_rate;
        for _ in 0..samples {
            data.push(f(t));
            t += sample_gap;
        }
        Signal::from_vec(data)
    }
}