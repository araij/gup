use clap::Parser;
use env_logger::{Builder, Env};
use gup::file::*;
use gup::graph::is_isomorphic_embedding;
use gup::gup14::{Gup, SearchOptions};
use itertools::repeat_n;
use itertools::Itertools;
use serde::Serialize;
use std::io::Write;
use std::time::{Duration, Instant};

/// Implementaton of the Guard-based Pruning (GuP) subgraph matching algorithm
#[derive(Parser, Debug, Serialize)]
#[command(author, version, about)]
struct Args {
    /// Number of times to repeat processing the same query set (for profiling)
    #[arg(long, default_value_t = 1)]
    repeat: usize,

    /// Base path of data graph files (.edges and .labels)"
    #[arg(short, long)]
    graph: String,

    /// Print detailed execution profiles
    #[arg(short, long, default_value_t = SearchOptions::default().probe)]
    probe: bool,

    /// Maximum number of matches to find
    #[arg(short = 'M', long, default_value_t = SearchOptions::default().match_limit)]
    match_limit: usize,

    /// Timeout for each query in seconds
    #[arg(short, long, default_value_t = SearchOptions::default().timeout.as_secs())]
    timeout: u64,

    /// Number of threads
    #[arg(long, default_value_t = SearchOptions::default().parallelism)]
    parallelism: usize,

    /// Maximum size of each reservation guard ('0' disables reservation guards)
    #[arg(short, long, default_value_t = SearchOptions::default().reservation_size)]
    reservation_size: usize,

    /// Disable nogood guards on vertices
    #[arg(long, default_value_t = SearchOptions::default().no_vertex_nogood)]
    no_vertex_nogood: bool,

    /// Disable nogood guards on edges
    #[arg(long, default_value_t = SearchOptions::default().no_edge_nogood)]
    no_edge_nogood: bool,

    /// Disable backjumping
    #[arg(long, default_value_t = SearchOptions::default().no_backjumping)]
    no_backjumping: bool,

    /// Query-set file
    query_set: String,
}

fn init_logger() {
    // Print ThreadID (it may differ `from rayon::current_thread_index()`)
    Builder::from_env(Env::default().default_filter_or("info"))
        .format(|buf, record| {
            let ts = buf.timestamp_micros();
            writeln!(
                buf,
                "[{} {:?} {} {}] {}",
                ts,
                std::thread::current().id(),
                record.level(),
                record.module_path().unwrap_or(""),
                record.args()
            )
        })
        .init();
}

fn print_result(n_matches: usize, search_sec: f32) {
    println!(
        "  \
  match_count: {n_matches}
  search_sec: {search_sec}",
    );
}

fn main() {
    init_logger();

    let args = Args::parse();

    println!("---");
    println!("command: {}", std::env::args().join(" "));
    serde_yaml::to_writer(std::io::stdout(), &args).unwrap();

    let (g, _, lmap) = read_renumbered_graph(&args.graph);
    let qs = read_queries(&args.query_set, &lmap);

    let opt = SearchOptions {
        probe: args.probe,
        match_limit: args.match_limit,
        timeout: Duration::from_secs(args.timeout),
        parallelism: args.parallelism,
        reservation_size: args.reservation_size,
        no_vertex_nogood: args.no_vertex_nogood,
        no_edge_nogood: args.no_edge_nogood,
        no_backjumping: args.no_backjumping,
    };
    let gup = Gup::new(&g, opt);

    let wholestart = Instant::now();
    println!("results:");

    for (i, q) in repeat_n(qs.iter().enumerate(), args.repeat).flatten() {
        println!("- index: {}", i);

        let t = Instant::now();
        let (nmatch, _) = gup.search(&q, |m| {
            debug_assert!(is_isomorphic_embedding(q, &g, &m.to_vec()).is_ok());
            true
        });
        print_result(nmatch, t.elapsed().as_secs_f32());
    }

    println!("whole_sec: {}", wholestart.elapsed().as_secs_f32());
}
