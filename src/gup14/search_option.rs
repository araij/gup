use std::time::Duration;

#[derive(Clone)]
pub struct SearchOptions {
    pub probe: bool,
    pub match_limit: usize,
    pub timeout: Duration,
    pub parallelism: usize,
    pub reservation_size: usize,
    pub no_vertex_nogood: bool,
    pub no_edge_nogood: bool,
    pub no_backjumping: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions {
            probe: false,
            match_limit: 100000,
            timeout: Duration::from_secs(u64::MAX),
            parallelism: 1,
            reservation_size: 3,
            no_vertex_nogood: false,
            no_edge_nogood: false,
            no_backjumping: false,
        }
    }
}
