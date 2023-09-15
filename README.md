GuP
===

Welcome to the repository of GuP, an efficient subgraph matching algorithm powered by guard-based pruning.
GuP is presented in the following paper:

- Junya Arai, Yasuhiro Fujiwara, and Makoto Onizuka. 2023. **GuP: Fast Subgraph Matching by Guard-based Pruning**. *Proceedings of ACM Management of Data* 1, 2, Article 167 (June 2023), 26 pages. https://doi.org/10.1145/3589312

Overview
--------

This repository currently contains a compiled executable of GuP for Linux, aimed to facilitate performance evaluations and comparisons. We will release the source code after refactoring for enhanced clarity and understanding of the algorithm.

:warning:**WARNING**  
All content within this repository is intended solely for evaluation purposes. Redistribution of any kind is not permitted. For more information, refer to the [`LICENSE`](./LICENSE) file.

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
./gup --graph examples/input/data_graph examples/input/query_set.yaml
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
$ ./gup --graph examples/inputs/data_graph examples/inputs/query_set.yaml
---
command: ./gup --graph examples/inputs/data_graph examples/inputs/query_set.yaml
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
  probe:
  match_count: 1
  search_sec: 0.000039965
  timed_out: false
[...]
whole_sec: 0.000083233
```

The average query processing time is given by `while_sec / <#query graphs>`.

:memo:**NOTE**  
`results[i].search_sec` may not be accurate for query graphs with very short execution times; please use `whole_sec` for calculating the average time.

If you require information about the number of recursive calls, you can use the `--probe` flag. This flag provides additional details about the search under the `probe` key, including `recursion_count`. Here's an example:

```bash
$ ./gup --probe --graph examples/inputs/data_graph examples/inputs/query_set.yaml
[...]
results:
- index: 0
  probe:
    gcs_vertex_count: 3
    gcs_edge_count: 4
    reservation_total: 0
    reservation_max: 0
    recursion_count: 4
    futile_recursion_count: 0
    backjump_count: 0
    removed_edge_count: 0
    naive_local_candidate_count: 3
    total_bytes_reservation: 144
    total_bytes_vertex_nogood: 168
    total_bytes_edge_nogood: 424
    optimized_total_bytes_edge_nogood: 0
  match_count: 1
  search_sec: 0.000060268
  timed_out: false
[...]
```
