GuP
===

Welcome to the repository of GuP, an efficient subgraph matching algorithm powered by guard-based pruning.
GuP is presented in the following paper:

- Junya Arai, Yasuhiro Fujiwara, and Makoto Onizuka. 2023. **GuP: Fast Subgraph Matching by Guard-based Pruning**. *Proceedings of ACM Management of Data* 1, 2, Article 167 (June 2023), 26 pages. https://doi.org/10.1145/3589312

:warning:**WARNING**  
All content within this repository is intended solely for evaluation purposes. Redistribution of any kind is not permitted. For more information, refer to the [`LICENSE`](./LICENSE) file.

How to Build
------------

1. Install Rust: <https://www.rust-lang.org/tools/install>.
2. Run `cargo build --release` in the directory where this repository is cloned.
    - The binary is generated in `./target/release/`

Example commands for Linux:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone https://github.com/araij/gup.git
cd gup
cargo build --release
```

Usage
-----

```bash
gup [OPTIONS] --graph <GRAPH> <QUERY_SET>
```

- A data graph is defined with a pair of `<GRAPH>.vertices` and `<GRAPH>.edges` files. Specify the path without the extensions when using the `--graph` option.
- A query set (`<QUERY_SET>`) is a YAML file containing definitions of query graphs.
- For an overview of `[OPTIONS]`, execute the command `gup --help`.

#### Example

```bash
target/release/gup --graph examples/inputs/data_graph examples/inputs/query_set.yaml
```

This command uses the data graph defined by the files `examples/input/data_graph.vertices` and `examples/input/data_graph.edges`.

## Inputs

For examples, refer to the files in [`examples/inputs/`](examples/inputs/).

### Data Graph

Data graph files, namely `.vertices` and `.edges`, are tab-separated value (TSV) files.

- **`.vertices`**: Represents vertex labels. Each line follows this format:
  ```tsv
  <vertex ID>	<label ID>
  ```
- **`.edges`**: Represents edges between vertices. Each line follows this format:
  ```tsv
  <source vertex ID>	<target vertex ID>
  ```

:memo:**NOTE**  
Our implementation assumes all edges to be undirected. Consequently, source and target vertex IDs are interchangeable.

### Query Set

A query-set file is a YAML file structured as follows:

```yaml
# Query graph #0
- vertices:
    - <label ID of vertex 0>
    - <label ID of vertex 1>
    - ...
  edges:
    - [<source vertex ID>, <target vertex ID>]
    - [<source vertex ID>, <target vertex ID>]
    - ...
# Query graph #1, and so on...
```

## Outputs

GuP prints a YAML-formatted output as shown below:

```bash
$ target/release/gup --graph examples/input/data_graph examples/input/query_set.yaml
---
command: target/release/gup --graph examples/inputs/data_graph examples/inputs/query_set.yaml
repeat: 1
graph: examples/inputs/data_graph
probe: false
match_limit: 100000
timeout: 18446744073709551615
parallelism: 1
reservation_size: 3
no_vertex_nogood: false
no_edge_nogood: false
no_backjumping: false
query_set: examples/inputs/query_set.yaml
results:
- index: 0
  match_count: 1
  search_sec: 0.000171016
- index: 1
[...]
whole_sec: 0.000394249
```

The average query processing time is given by `while_sec / <#query graphs>`.

:memo:**NOTE**  
`results[...].search_sec` may not be accurate for query graphs with very short execution times; please use `whole_sec` for calculating the average time.

If you need to count the number of recursive calls, consider using the `--probe` flag. This flag introduces a `probe` key in the YAML output, detailing internal counters about the search process, as shown in the example below. Avoid using this flag for performance measurements due to its additional computational overhead.

```bash
$ target/release/gup --probe --graph examples/inputs/data_graph examples/inputs/query_set.yaml
---
command: target/release/gup --probe --graph examples/inputs/data_graph examples/inputs/query_set.yaml
repeat: 1
graph: examples/inputs/data_graph
probe: true
match_limit: 100000
timeout: 18446744073709551615
parallelism: 1
reservation_size: 3
no_vertex_nogood: false
no_edge_nogood: false
no_backjumping: false
query_set: examples/inputs/query_set.yaml
results:
- index: 0
  probe:
    matching_order: [0, 1, 2]
    gcs_vertex_count: 3
    gcs_edge_count: 4
    reservation_total: 0
    reservation_max: 0
    recursion_count: 4
    futile_recursion_count: 0
    guard_count_reservation: 0
    guard_count_vertex_nogood: 0
    guard_count_edge_nogood: 0
    backjump_count: 0
    total_bytes_reservation: 144
    total_bytes_vertex_nogood: 144
    total_bytes_edge_nogood: 0
    threads:
    - running_secs: 0.000003122
  match_count: 1
  search_sec: 0.000266643
[...]
```
