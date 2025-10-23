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

pub fn draw_rounds_chart(
    vec_lens: &[usize],
    batch_ms: &[f64],
    tiling_ms: &[f64],
    t: u32,
    d: u32,
    threads: usize,
    tile_len: usize,
    out_path: &str,
) {
    let out_rounds = std::path::Path::new(out_path).with_file_name("bench_by_rounds.png");
    let root = BitMapBackend::new(out_rounds.to_str().unwrap(), (1280, 720)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    // X-axis as T = log2(vec_len), linear scale
    let ts: Vec<i32> = vec_lens.iter().map(|&n| if n == 0 { 0 } else { (usize::BITS as usize - 1 - n.leading_zeros() as usize) as i32 }).collect();
    let xmin = *ts.iter().min().unwrap_or(&0);
    let xmax = *ts.iter().max().unwrap_or(&0);
    // Compute per-round speedup and clamp for log scale
    let mut points: Vec<(i32, f64)> = Vec::new();
    for i in 0..batch_ms.len() {
        let tval = if i < tiling_ms.len() { tiling_ms[i] } else { 0.0 };
        let b = batch_ms[i];
        let tt = ts.get(i).copied().unwrap_or(0);
        if tt >= 5 && tval > 0.0 { points.push((tt, (b / tval).max(1e-3))); }
    }
    let mut ymax = points.iter().map(|p| p.1).fold(0.0, f64::max) * 1.2;
    if ymax < 2.0 { ymax = 2.0; }
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Per-round Speedup (batch/tiling, log) — d={}, T={}, thr={}, tile_len={}", d, t, threads, tile_len),
            ("sans-serif", 24)
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d((xmin - 1)..(xmax + 1), (1e-3..ymax).log_scale())
        .unwrap();
    chart.configure_mesh().x_desc("T (vec_len = 2^T)").y_desc("speedup (log)").draw().unwrap();
    chart.draw_series(LineSeries::new(points, &BLACK)).unwrap().label("speedup").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw().unwrap();
    root.present().unwrap();
}

pub fn draw_rounds_speedup_chart(
    vec_lens: &[usize],
    batch_ms: &[f64],
    tiling_ms: &[f64],
    out_path: &str,
) {
    let out_rounds = std::path::Path::new(out_path).with_file_name("bench_by_rounds_speedup.png");
    let root = BitMapBackend::new(out_rounds.to_str().unwrap(), (1280, 720)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let ts: Vec<i32> = vec_lens.iter().map(|&n| if n == 0 { 0 } else { (usize::BITS as usize - 1 - n.leading_zeros() as usize) as i32 }).collect();
    let points: Vec<(i32, f64)> = (0..batch_ms.len())
        .filter_map(|i| {
            let tval = *tiling_ms.get(i).unwrap_or(&0.0);
            let b = *batch_ms.get(i).unwrap_or(&0.0);
            let tt = *ts.get(i).unwrap_or(&0);
            if tt >= 8 && tval > 0.0 { Some((tt, b / tval)) } else { None }
        })
        .collect();
    let xmin = points.iter().map(|p| p.0).min().unwrap_or(0);
    let xmax = points.iter().map(|p| p.0).max().unwrap_or(0);
    let mut ymin = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    if !ymin.is_finite() { ymin = 0.5; }
    ymin = ymin.min(0.9); // ensure lower bound shows slowdowns
    let ymax = points.iter().map(|p| p.1).fold(0.0, f64::max) * 1.2;
    let mut chart = ChartBuilder::on(&root)
        .caption("Per-round Speedup (batch/tiling)", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d((xmin - 1)..(xmax + 1), (ymin.max(0.5)..ymax.max(1.5)).log_scale())
        .unwrap();
    chart.configure_mesh().x_desc("T (vec_len = 2^T)").y_desc("speedup (log)").draw().unwrap();
    chart.draw_series(LineSeries::new(points, &BLACK)).unwrap().label("speedup").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw().unwrap();
    root.present().unwrap();
}

pub fn draw_rounds_normalized_chart(
    vec_lens: &[usize],
    batch_ms: &[f64],
    tiling_ms: &[f64],
    out_path: &str,
) {
    let out_norm = std::path::Path::new(out_path).with_file_name("bench_by_rounds_norm.png");
    let root = BitMapBackend::new(out_norm.to_str().unwrap(), (1280, 720)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let ts: Vec<i32> = vec_lens.iter().map(|&n| if n == 0 { 0 } else { (usize::BITS as usize - 1 - n.leading_zeros() as usize) as i32 }).collect();
    // Normalized speedup is identical to raw speedup; plot speedup with log y
    let floor = 1e-3f64;
    let points: Vec<(i32, f64)> = (0..ts.len()).filter_map(|i| {
        let b = *batch_ms.get(i).unwrap_or(&0.0);
        let t = *tiling_ms.get(i).unwrap_or(&0.0);
        if ts[i] >= 8 && t > 0.0 { Some((ts[i], (b / t).max(floor))) } else { None }
    }).collect();
    let mut ymax = points.iter().map(|p| p.1).fold(0.0, f64::max) * 1.5;
    if ymax < 2.0 { ymax = 2.0; }
    let xmin = *ts.iter().min().unwrap_or(&0);
    let xmax = *ts.iter().max().unwrap_or(&0);
    let mut chart = ChartBuilder::on(&root)
        .caption("Per-round Speedup (log)", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d((xmin - 1)..(xmax + 1), (floor..ymax).log_scale())
        .unwrap();
    chart.configure_mesh().x_desc("T (vec_len = 2^T)").y_desc("speedup (log)").draw().unwrap();
    chart.draw_series(LineSeries::new(points, &BLACK)).unwrap().label("speedup").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw().unwrap();
    root.present().unwrap();
}

pub fn draw_rounds_tail_chart(
    vec_lens: &[usize],
    batch_ms: &[f64],
    tiling_ms: &[f64],
    last_n: usize,
    out_path: &str,
) {
    let out_tail = std::path::Path::new(out_path).with_file_name("bench_by_rounds_tail.png");
    let root = BitMapBackend::new(out_tail.to_str().unwrap(), (1280, 720)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut data: Vec<(i32, f64)> = vec_lens.iter().enumerate().filter_map(|(i, &n)| {
        let t = if n == 0 { 0 } else { (usize::BITS as usize - 1 - n.leading_zeros() as usize) as i32 };
        let b = *batch_ms.get(i).unwrap_or(&0.0);
        let ti = *tiling_ms.get(i).unwrap_or(&0.0);
        if t >= 8 && ti > 0.0 {
            let sp = (b / ti).max(1e-3);
            Some((t, sp))
        } else { None }
    }).collect();
    data.sort_by_key(|(t, _)| *t);
    let take = last_n.min(data.len());
    let tail = &data[data.len().saturating_sub(take)..];
    let xmin = tail.iter().map(|p| p.0).min().unwrap_or(0);
    let xmax = tail.iter().map(|p| p.0).max().unwrap_or(0);
    let ymax = tail.iter().map(|p| p.1).fold(0.0, f64::max) * 1.2;
    let mut chart = ChartBuilder::on(&root)
        .caption("Per-round Speedup (largest rounds, log)", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d((xmin - 1)..(xmax + 1), (1e-3..ymax).log_scale())
        .unwrap();
    chart.configure_mesh().x_desc("T (vec_len = 2^T)").y_desc("speedup (log)").draw().unwrap();
    let series_sp: Vec<(i32, f64)> = tail.iter().map(|(t, sp)| (*t, *sp)).collect();
    chart.draw_series(LineSeries::new(series_sp, &BLACK)).unwrap().label("speedup").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw().unwrap();
    root.present().unwrap();
}

pub fn draw_rounds_speedup_multi(
    series: &[(String, Vec<i32>, Vec<f64>)],
    out_path: &str,
) {
    let out_multi = std::path::Path::new(out_path).with_file_name("bench_by_rounds_speedup.png");
    let root = BitMapBackend::new(out_multi.to_str().unwrap(), (1280, 720)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    // Determine bounds
    let xmin = series.iter().flat_map(|(_, xs, _)| xs.iter().copied()).min().unwrap_or(0);
    let xmax = series.iter().flat_map(|(_, xs, _)| xs.iter().copied()).max().unwrap_or(0);
    let floor = 1e-3f64;
    let mut ymax = series.iter().flat_map(|(_, _, ys)| ys.iter().copied()).fold(0.0, f64::max) * 1.2;
    if ymax < 2.0 { ymax = 2.0; }
    let mut chart = ChartBuilder::on(&root)
        .caption("Per-round Speedup (all runs, log)", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d((xmin - 1)..(xmax + 1), (floor..ymax).log_scale())
        .unwrap();
    chart.configure_mesh().x_desc("T (vec_len = 2^T)").y_desc("speedup (log)").draw().unwrap();

    // Simple color palette
    let palette: [RGBColor; 7] = [RED, GREEN, BLUE, MAGENTA, CYAN, BLACK, YELLOW];
    for (i, (label, xs, ys)) in series.iter().enumerate() {
        let col = palette[i % palette.len()];
        let style = ShapeStyle { color: col.to_rgba(), filled: false, stroke_width: 2 };
        let pts: Vec<(i32, f64)> = xs.iter().copied().zip(ys.iter().copied()).filter(|(tt, v)| *tt >= 8 && *v > 0.0).collect();
        if !pts.is_empty() {
            chart.draw_series(LineSeries::new(pts, style)).unwrap()
                .label(label.clone())
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], ShapeStyle { color: col.to_rgba(), filled: false, stroke_width: 2 }));
        }
    }
    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw().unwrap();
    root.present().unwrap();
}


