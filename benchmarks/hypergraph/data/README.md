# Hypergraph benchmark inputs

These files are pinned local copies used by the reproducible IEP validation.
The validation records and checks their SHA-256 digests before evaluating any
saved assignment.

| file | source | SHA-256 |
|---|---|---|
| `hmetis/ibm01.hgr` | mt-KaHyPar examples at commit `eee7b7a03dbbd565a39f6cb2679b083e484c905e` | `40f7f7c4dfd96c06b0570f696e67b9c667ac2cbdf4d0858da5e690f4f4d5ac72` |
| `hmetis/ibm02.hgr` | TILOS HypergraphPartitioning at commit `ff614601a9b8f7853e21019e1fd320f44e445f3c` | `ff09f3be9ed84a8c13257f1655555938072cdf01fae40f1548795763981eae05` |
| `hmetis/bad_for_ec.hgr` | small edge-coarsening pathology fixture, attributed in the file to `[KarKu22000]` | `33ddff8bb4c3f9ccbb36a2df1deb8184c5157d2d340a9e39a952e237caf56416` |

The IBM files are unweighted hMETIS instances. `bad_for_ec.hgr` is a small
sanity fixture and is not included in aggregate real-instance performance
claims.
