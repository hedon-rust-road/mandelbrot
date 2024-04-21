use image::png::PNGEncoder;
use image::ColorType;
use num::Complex;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;
use std::env;
use std::fs::File;
use std::str::FromStr;

/// 尝试测试 `c` 是否位于曼德博集中，使用最多 `limit` 次迭代来判定
///
/// 如果 `c` 不是集合成员之一，则返回 `Some(i)`，其中 `i` 是 `c` 离开以原点
/// 为中心的半径为 2 的圆时所需的迭代次数。如果 `c` 似乎是集群成员之一（确
/// 切而言是达到了迭代次数限制但仍然无法证明 `c` 不是成员），则返回 `None`
fn escape_time(c: Complex<f64>, limit: usize) -> Option<usize> {
    let mut z = Complex { re: 0.0, im: 0.0 };
    for i in 0..limit {
        if z.norm_sqr() > 4.0 {
            return Some(i);
        }
        z = z * z + c
    }
    None
}

/// 把字符串 `s`（形如 `"400×600"` 或 ``"1.0,0.5"）解析成一个坐标对
///
/// 具体来说，`s` 应该具有<left><sep><right>的格式，其中<sep>是由`separator`
/// 参数给出的字符，而<left>和<right>是可以被 `T:from_str` 解析的字符串。
/// `separator` 必须是 ASCII 字符
///
/// 如果 `s` 具有正确的格式，就返回 `Some(x,y)`，否则返回 `None`
fn parse_pair<T: FromStr>(s: &str, separator: char) -> Option<(T, T)> {
    match s.find(separator) {
        None => None,
        Some(index) => match (T::from_str(&s[..index]), T::from_str(&s[index + 1..])) {
            (Ok(l), Ok(r)) => Some((l, r)),
            _ => None,
        },
    }
}

#[test]
fn test_parse_pair() {
    assert_eq!(parse_pair::<i32>("", ','), None);
    assert_eq!(parse_pair::<i32>("10,", ','), None);
    assert_eq!(parse_pair::<i32>(",10", ','), None);
    assert_eq!(parse_pair::<i32>("10,20", ','), Some((10, 20)));
    assert_eq!(parse_pair::<i32>("10,20xy", ','), None);
    assert_eq!(parse_pair::<f64>("0.5x", 'x'), None);
    assert_eq!(parse_pair::<f64>("0.5x1.5", 'x'), Some((0.5, 1.5)));
}

/// 把一对用逗号隔开的浮点数解析为复数
fn parse_complex(s: &str) -> Option<Complex<f64>> {
    match parse_pair(s, ',') {
        Some((re, im)) => Some(Complex { re, im }),
        None => None,
    }
}

#[test]
fn test_parse_complex() {
    assert_eq!(
        parse_complex("1.25,-0.0625"),
        Some(Complex {
            re: 1.25,
            im: -0.0625
        })
    );
    assert_eq!(parse_complex(",-0.0625"), None);
}

/// 给定输出图像重像素的行和列，返回复平面中对应的坐标
///
/// `bound` 是一个 `pair`，给出了图像的像素宽度和像素高度。
/// `pixed` 是表示给图片中特定像素的 (column, row) 二元组。
/// `upper_left` 参数和 `lower_right` 参数是在复平面中表示指定图像覆盖范围的点。
fn pixed_to_point(
    /*
    ·--------------------> bounds.0  re
    丨
    丨
    丨
    丨
    丨
    bounds.1  im
     */
    bounds: (usize, usize),
    pixed: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) -> Complex<f64> {
    let (width, height) = (
        lower_right.re - upper_left.re, // 右-左
        upper_left.im - lower_right.im, // 上-下
    );

    Complex {
        re: upper_left.re + pixed.0 as f64 * width / bounds.0 as f64,
        im: upper_left.im - pixed.1 as f64 * height / bounds.1 as f64,
    }
}

#[test]
fn test_pixed_to_point() {
    assert_eq!(
        pixed_to_point(
            (100, 200),
            (25, 175),
            Complex { re: -1.0, im: 1.0 },
            Complex { re: 1.0, im: -1.0 }
        ),
        Complex {
            re: -0.5,
            im: -0.75,
        }
    );
}

/// 将曼德博集对应的矩形渲染到像素缓冲区中
///
/// `bounds` 参数会给缓冲区 `pixels` 的宽度和高度，此缓冲区的每个字节都
/// 包含一个灰度像素。`upper_left` 和 `lower_right` 参数分别指定了
/// 复平面中对应于像素缓冲区左上角和右上角的点。
fn render(
    pixels: &mut [u8],
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) {
    assert_eq!(pixels.len(), bounds.0 * bounds.1);

    for raw in 0..bounds.1 {
        for column in 0..bounds.0 {
            let point = pixed_to_point(bounds, (column, raw), upper_left, lower_right);
            pixels[raw * bounds.0 + column] = match escape_time(point, 255) {
                None => 0,
                Some(count) => 255 - count as u8,
            }
        }
    }
}

/// 把 `pixels` 缓冲区（其尺寸由 `bounds` 给出）写入名为 `filename` 的文件中
fn write_image(
    filename: &str,
    pixels: &[u8],
    bounds: (usize, usize),
) -> Result<(), std::io::Error> {
    let output = File::create(filename)?;
    let encoder = PNGEncoder::new(output);
    encoder.encode(pixels, bounds.0 as u32, bounds.1 as u32, ColorType::Gray(8))?;
    Ok(())
}

/// 单线程
/// ➜  mandelbrot git:(master) ✗ time target/release/mandelbrot mandel.png 4000x3000 -1.20,0.35 -1,0.20
/// target/release/mandelbrot mandel.png 4000x3000 -1.20,0.35 -1,0.20  3.30s user 0.01s system 97% cpu 3.372 total
/// 多线程
/// ➜  mandelbrot git:(master) ✗ time target/release/mandelbrot mandel2.png 4000x3000 -1.20,0.35 -1,0.20
/// target/release/mandelbrot mandel2.png 4000x3000 -1.20,0.35 -1,0.20  6.34s user 0.01s system 553% cpu 1.148 total
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!("Usage: {} FILE PIXELS UPPERLEFT LOWERRIGHT", args[0]);
        eprintln!(
            "Example: {} mandel.png 1000x700 -1.20,0.35 -1,0.20",
            args[0]
        );
        std::process::exit(1);
    }

    let bounds = parse_pair(&args[2], 'x').expect("error parsing image dimensions");
    let upper_left = parse_complex(&args[3]).expect("error parsing upper left corner point");
    let lower_right = parse_complex(&args[4]).expect("error parsing lower right corner point");
    let mut pixels = vec![0; bounds.0 * bounds.1];

    // /*
    // ③ rayon 窃取式并行
    let bands: Vec<(usize, &mut [u8])> = pixels.chunks_mut(bounds.0).enumerate().collect();

    bands.into_par_iter().for_each(|(i, band)| {
        let top = i;
        let band_bounds = (bounds.0, 1);
        let band_upper_left = pixed_to_point(bounds, (0, top), upper_left, lower_right);
        let band_lower_right = pixed_to_point(bounds, (bounds.0, top + 1), upper_left, lower_right);
        render(band, band_bounds, band_upper_left, band_lower_right);
    });
    // */
    /*
    // ① 单线程执行
    // render(&mut pixels, bounds, upper_left, lower_right);
     */

    /*
    // ② 并发执行
    let threads = 8;
    let rows_per_band = bounds.1 / threads + 1;
    {
        let bands: Vec<&mut [u8]> = pixels.chunks_mut(rows_per_band * bounds.0).collect();
        crossbeam::scope(|spawner| {
            for (i, band) in bands.into_iter().enumerate() {
                let top = rows_per_band * i;
                let height = band.len() / bounds.0;
                let band_bounds = (bounds.0, height);
                let band_upper_left = pixed_to_point(bounds, (0, top), upper_left, lower_right);
                let band_lower_right =
                    pixed_to_point(bounds, (bounds.0, top + height), upper_left, lower_right);
                spawner.spawn(move |_| {
                    render(band, band_bounds, band_upper_left, band_lower_right);
                });
            }
        })
        .unwrap();
    }
     */
    write_image(&args[1], &pixels, bounds).expect("error writing PNG file");
}
