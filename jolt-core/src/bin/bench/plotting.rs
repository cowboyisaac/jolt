use plotters::prelude::*;

pub type Row = (u32, u32, usize, usize, f64, f64, f64, f64, usize, f64, f64, f64, f64, f64, f64);

pub fn draw_all_charts(
    rows: &[Row],
    t_list: &[u32],
    d_list: &[u32],
    thread_variants: &[usize],
    out_path: &str,
) {
    // Chart 1: by trace size (T)
    let root = BitMapBackend::new(out_path, (1280, 720)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let y_max = rows.iter()
        .map(|r| r.9.max(r.10).max(r.11).max(r.12))
        .fold(0.0, f64::max) * 1.2;
    let t_min = *t_list.iter().min().unwrap_or(&0) as i32;
    let t_max = *t_list.iter().max().unwrap_or(&0) as i32;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Sumcheck Proving Time (ms) — threads={:?}", thread_variants),
            ("sans-serif", 24)
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d((t_min - 1)..(t_max + 1), 0f64..y_max)
        .unwrap();
    chart.configure_mesh().x_desc("T").y_desc("ms").draw().unwrap();

    for &d in d_list {
        for &thr in thread_variants {
            let mut series_batch_first: Vec<(i32, f64)> = Vec::new();
            let mut series_batch_prove: Vec<(i32, f64)> = Vec::new();
            let mut series_tiling_first: Vec<(i32, f64)> = Vec::new();
            let mut series_tiling_prove: Vec<(i32, f64)> = Vec::new();
            for &(t, dd, threads_here, _tile_len, _gen, _pb, _pt, _tot, _thr_dup, ib, cb, it, ct, _ibatch, _itiling) in rows.iter() {
                if dd == d && threads_here == thr {
                    series_batch_first.push((t as i32, ib));
                    series_batch_prove.push((t as i32, cb));
                    series_tiling_first.push((t as i32, it));
                    series_tiling_prove.push((t as i32, ct));
                }
            }
            chart
                .draw_series(LineSeries::new(series_batch_first, &RED))
                .unwrap()
                .label(format!("batch boot-kernel d={} thr={}", d, thr))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
            chart
                .draw_series(LineSeries::new(series_batch_prove, &GREEN))
                .unwrap()
                .label(format!("batch recursive-kernel d={} thr={}", d, thr))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
            chart
                .draw_series(LineSeries::new(series_tiling_first, &BLUE))
                .unwrap()
                .label(format!("tiling boot-kernel d={} thr={}", d, thr))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
            chart
                .draw_series(LineSeries::new(series_tiling_prove, &MAGENTA))
                .unwrap()
                .label(format!("tiling recursive-kernel d={} thr={}", d, thr))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));
        }
    }
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
    root.present().unwrap();

    // Chart 2: by threads
    let out_threads = std::path::Path::new(out_path).with_file_name("bench_by_threads.png");
    let root2 = BitMapBackend::new(out_threads.to_str().unwrap(), (1280, 720)).into_drawing_area();
    root2.fill(&WHITE).unwrap();
    let threads_min = *thread_variants.iter().min().unwrap_or(&1) as i32;
    let threads_max = *thread_variants.iter().max().unwrap_or(&1) as i32;
    let y2_max = rows.iter().map(|r| r.9.max(r.10).max(r.11).max(r.12)).fold(0.0, f64::max) * 1.2;
    let mut chart2 = ChartBuilder::on(&root2)
        .caption(
            format!("Sumcheck Proving Time (ms) — by threads, Ts={:?}", t_list),
            ("sans-serif", 24)
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d((threads_min - 1)..(threads_max + 1), 0f64..y2_max)
        .unwrap();
    chart2.configure_mesh().x_desc("threads").y_desc("ms").draw().unwrap();

    for &d in d_list {
        for &t in t_list {
            let mut b_first: Vec<(i32, f64)> = Vec::new();
            let mut b_prove: Vec<(i32, f64)> = Vec::new();
            let mut t_first: Vec<(i32, f64)> = Vec::new();
            let mut t_prove: Vec<(i32, f64)> = Vec::new();
            for &(tt, dd, threads_here, _tile_len, _gen, _pb, _pt, _tot, _thr_dup, ib, cb, it, ct, _ibatch, _itiling) in rows.iter() {
                if dd == d && tt == t {
                    b_first.push((threads_here as i32, ib));
                    b_prove.push((threads_here as i32, cb));
                    t_first.push((threads_here as i32, it));
                    t_prove.push((threads_here as i32, ct));
                }
            }
            chart2.draw_series(LineSeries::new(b_first, &RED)).unwrap()
                .label(format!("batch boot-kernel d={} T={}", d, t))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
            chart2.draw_series(LineSeries::new(b_prove, &GREEN)).unwrap()
                .label(format!("batch recursive-kernel d={} T={}", d, t))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
            chart2.draw_series(LineSeries::new(t_first, &BLUE)).unwrap()
                .label(format!("tiling boot-kernel d={} T={}", d, t))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
            chart2.draw_series(LineSeries::new(t_prove, &MAGENTA)).unwrap()
                .label(format!("tiling recursive-kernel d={} T={}", d, t))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));
        }
    }
    chart2
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
    root2.present().unwrap();

    // Chart 3: by tile_len
    let out_tile = std::path::Path::new(out_path).with_file_name("bench_by_tile_len.png");
    let root3 = BitMapBackend::new(out_tile.to_str().unwrap(), (1280, 720)).into_drawing_area();
    root3.fill(&WHITE).unwrap();
    // Only consider tiling series for tile_len chart
    let tiling_rows: Vec<&Row> = rows.iter().filter(|r| r.3 != 0).collect();
    let tile_min = tiling_rows.iter().map(|r| r.3).min().unwrap_or(0) as i32;
    let tile_max = tiling_rows.iter().map(|r| r.3).max().unwrap_or(0) as i32;
    let y3_max = tiling_rows.iter().map(|r| r.11).fold(0.0, f64::max) * 1.2;
    let mut chart3 = ChartBuilder::on(&root3)
        .caption("Sumcheck Proving Time (ms) — by tile_len", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d((tile_min - 1)..(tile_max + 1), 0f64..y3_max)
        .unwrap();
    chart3.configure_mesh().x_desc("tile_len").y_desc("ms").draw().unwrap();
    for &d in d_list {
        for &thr in thread_variants {
            let mut t_first: Vec<(i32, f64)> = Vec::new();
            let mut t_prove: Vec<(i32, f64)> = Vec::new();
            for &(_t, dd, threads_here, tile_len, _gen, _pb, _pt, _tot, _thr_dup, _ib, _cb, it, ct, _ibatch, _itiling) in rows.iter() {
                if dd == d && threads_here == thr && tile_len != 0 {
                    t_first.push((tile_len as i32, it));
                    t_prove.push((tile_len as i32, ct));
                }
            }
            chart3.draw_series(LineSeries::new(t_first, &BLUE)).unwrap().label(format!("tiling boot-kernel d={} thr={}", d, thr)).legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
            chart3.draw_series(LineSeries::new(t_prove, &MAGENTA)).unwrap().label(format!("tiling recursive-kernel d={} thr={}", d, thr)).legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));
        }
    }
    chart3.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw().unwrap();
    root3.present().unwrap();
}


