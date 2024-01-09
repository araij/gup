use itertools::Itertools;
use serde::{Deserialize, Serialize};
use serde_yaml::Value;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::iter::once;
use std::path::Path;

use super::graph::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryElement {
    pub edges: Vec<(Vertex, Vertex)>,
    pub vertices: Vec<Label>,
}

pub type QueryFile = Vec<QueryElement>;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct QueryResult {
    pub index: usize,
    pub match_count: usize,
    pub search_sec: f32,
    pub probe: Option<Value>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct CliResult {
    pub command: String,
    pub graph: String,
    pub match_limit: usize,
    pub query_set: String,
    pub whole_sec: f32,
    pub results: Vec<QueryResult>,
}

pub fn open<P: AsRef<Path>>(path: P) -> File {
    File::open(&path)
        .unwrap_or_else(|_| panic!("Cannot open: {}", path.as_ref().display()))
}

/// Reads a file that each line represents a space-separated record.
fn read_parsed_tuples<T>(path: String) -> impl Iterator<Item = T>
where
    T: itertools::traits::HomogeneousTuple,
    T::Item: std::str::FromStr,
{
    // This function takes a path, not `BufReader`, as an argument to show
    // comprehensible error messages.
    BufReader::new(open(&path))
        .lines()
        .enumerate()
        .filter_map(move |(i, r)| {
            let s = r.unwrap_or_else(|_| panic!("Cannot read '{}'", &path));
            let s = s.trim();
            if s.len() == 0 || s.starts_with("#") {
                return None;
            }
            let t = s
                .split_whitespace()
                .map(|x| {
                    x.parse().unwrap_or_else(|_| {
                        panic!("Cannot parse line {} of '{}'", i, &path)
                    })
                })
                .collect_tuple()
                .unwrap_or_else(|| {
                    panic!(
                        "Invalid number of tokens at line {} of '{}'",
                        i, &path
                    )
                });
            Some(t)
        })
}

fn with_ext<P: AsRef<Path>, S: AsRef<OsStr>>(path: P, ext: S) -> String {
    path.as_ref()
        .with_extension(ext)
        .to_str()
        .unwrap_or_else(|| {
            panic!("Cannot convert a path to str: {}", path.as_ref().display())
        })
        .to_string()
}

pub fn read_edges<P: AsRef<Path>>(
    stem_path: P,
) -> impl Iterator<Item = (Vertex, Vertex)> {
    read_parsed_tuples(with_ext(stem_path, "edges"))
}

pub fn read_vertices<P: AsRef<Path>>(
    stem_path: P,
) -> impl Iterator<Item = (Vertex, Label)> {
    read_parsed_tuples(with_ext(stem_path, "vertices"))
}

fn present_vertices(edges: &Vec<Edge>) -> Vec<Vertex> {
    let mut vs: Vec<_> = edges
        .iter()
        .flat_map(|&(s, t)| once(s).chain(once(t)))
        .collect();
    vs.sort();
    vs.dedup();
    vs
}

/// Both `attrs` and `present_verts` must be sorted.
fn filter_present<'a, I: 'a + IntoIterator<Item = (Vertex, Label)>>(
    attrs: I,
    present_verts: &'a Vec<Vertex>,
) -> impl 'a + Iterator<Item = (Vertex, Label)> {
    // `attrs` also must be sorted, but cannot check it without consuming the
    // iterator...
    debug_assert!(present_verts.iter().tuple_windows().all(|(x, y)| x <= y));

    present_verts.iter().scan(attrs.into_iter(), |s, &v| {
        while let Some((u, l)) = s.next() {
            if u < v {
                continue;
            } else if u == v {
                return Some((u, l));
            } else {
                panic!("Cannot find a label for vertex {}", v);
            }
        }
        panic!("Cannot find a label for vertices of ID>={}", v);
    })
}

/// Reads a graph from files (`.vertices` and `.edges`)
///
/// Vertex IDs and Label IDs are renumbered so that they are contiguous and do
/// not includes unused IDs.
pub fn read_renumbered_graph<P: AsRef<Path>>(
    stem_path: P,
) -> (Graph, HashMap<Vertex, Vertex>, HashMap<Label, Label>) {
    let es: Vec<_> = read_edges(&stem_path).collect();
    let vs = present_vertices(&es);

    // Read vertex attributes, dropping those of absent vertices
    let vattrs: Vec<_> =
        filter_present(read_vertices(&stem_path).sorted(), &vs).collect();
    // Sorted labels without duplicates
    let mut ls: Vec<_> = vattrs.iter().map(|x| x.1).collect();
    ls.sort();
    ls.dedup();

    // Original vertex ID -> Renumbered vertex ID
    let vmap: HashMap<Vertex, Vertex> = vs
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as Vertex))
        .collect();
    // Original label ID -> Renumbered label ID
    let lmap: HashMap<Label, Label> = ls
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i as Label))
        .collect();

    let g = Graph::from_edges(
        vattrs.iter().map(|(_, l)| lmap[l]),
        es.iter().map(|(s, t)| (vmap[s], vmap[t])),
    );
    (g, vmap, lmap)
}

fn create_graph(
    q: &QueryElement,
    index: usize,
    lmap: &HashMap<Label, Label>,
) -> Graph {
    let vertices = q.vertices.iter().map(|l| {
        lmap.get(l).unwrap_or_else(|| {
            panic!(
                "Query #{index} contains label {l}, which does not appear in \
                    the data graph"
            )
        })
    });
    Graph::from_edges(vertices, &q.edges)
}

/// Returns a vector of queries.
pub fn read_queries(qfile: &str, lmap: &HashMap<Label, Label>) -> Vec<Graph> {
    serde_yaml::from_reader::<_, QueryFile>(open(qfile))
        .unwrap_or_else(|_| {
            panic!("Cannot read or cannot parse the query file: {}", qfile)
        })
        .iter()
        .enumerate()
        .map(|(i, q)| create_graph(q, i, lmap))
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use std::io::{Result, Write};

    #[test]
    fn test_read_parsed_tuples() -> Result<()> {
        let mut f = tempfile::NamedTempFile::new()?;
        // Includes
        // - Comments
        // - Empty lines
        // - Leading/trailing spaces
        // - Mixed tabs
        // - Spaces
        writeln!(
            &mut f,
            r#"
# comment
1 11    
# comment
        2 22 
3	33

4	    44
5	55      
"#,
        )?;
        f.flush()?;

        let it = read_parsed_tuples::<(i32, i32)>(
            f.path().to_str().unwrap().to_string(),
        );
        let it_expected = [(1, 11), (2, 22), (3, 33), (4, 44), (5, 55)];

        assert_eq!(it.eq(it_expected.iter().cloned()), true);
        Ok(())
    }

    #[test]
    fn test_read_renumbered_graph() -> Result<()> {
        //     v8 (3)
        //       /   \
        //  v2 (9)---(4) v7
        //
        let dir = tempfile::tempdir()?;
        let mut efile = File::create(dir.path().join("graph.edges"))?;
        let mut vfile = File::create(dir.path().join("graph.vertices"))?;

        // This test pattern include duplicate edges and loops
        writeln!(
            &mut efile,
            r#"
8 2
7 7
2 7
7 8
2 7
"#,
        )?;
        writeln!(
            &mut vfile,
            r#"
2 9
8 3
7 4
"#,
        )?;

        let (g, vmap, lmap) = read_renumbered_graph(dir.path().join("graph"));

        let g_expect = Graph::from_edges(&[2, 1, 0], &[(0, 1), (1, 2), (2, 0)]);
        assert_eq!(g, g_expect);
        assert_eq!(vmap, [(2, 0), (7, 1), (8, 2)].iter().cloned().collect());
        assert_eq!(lmap, [(3, 0), (4, 1), (9, 2)].iter().cloned().collect());

        Ok(())
    }
}
