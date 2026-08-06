# Preliminary offline component evaluation report

This report separates deterministic component conformance, contract-test counts, and local microbenchmarks. Fixed-set misses remain visible and are not integrity failures.

# Preliminary offline component evaluation

| Measure | Result |
|---|---:|
| Routing reason conformance | 33/60 (0.550) |
| Helper action conformance | 33/60 (0.550) |
| Direct action conformance | 26/43 (0.605) |
| Query conformance | 8/8 (1.000) |
| AquiLLM evidence macro recall | 16.666666666666668/22 (0.758) |
| Memory exact-set conformance | 30/40 (0.750) |
| Included contract tests passed | 68/68 |

Limitations: No generated-answer correctness, relevance, faithfulness, or citation-entailment claim. No end-to-end latency, concurrency, GPU, or production-throughput claim. No authorization or database-isolation claim from syntax and prefix checks. No population estimate or sampling-based confidence interval.

## Contract tests

- collected: 68
- passed: 68
- failed: 0
- skipped: 0
- errors: 0
- unavailable: 1

## Timings

| Module | Input size | Median seconds | p95 seconds | Throughput/s |
|---|---|---:|---:|---:|
| evidence | {"candidate_count": 1} | 0.000003300 | 0.000003800 | 303027.996 |
| evidence | {"candidate_count": 10} | 0.000011900 | 0.000019900 | 84033.385 |
| evidence | {"candidate_count": 100} | 0.000033400 | 0.000045400 | 29940.153 |
| memory | {"characters": 82} | 0.000021500 | 0.000042400 | 46511.652 |
| routing | {"characters": 48} | 0.000010900 | 0.000011500 | 91742.911 |

## Detailed aggregate metrics

```json
{
  "action": {
    "direct": {
      "by_label": {
        "local_tool_handling": {
          "conformance": {
            "denominator": 0,
            "numerator": 0,
            "status": "not_applicable",
            "value": null
          },
          "support": 0
        },
        "prompt_select_collection": {
          "conformance": {
            "denominator": 15,
            "numerator": 8,
            "status": "ok",
            "value": 0.5333333333333333
          },
          "support": 15
        },
        "retrieve": {
          "conformance": {
            "denominator": 14,
            "numerator": 8,
            "status": "ok",
            "value": 0.5714285714285714
          },
          "support": 14
        },
        "skip_normal_tool_loop": {
          "conformance": {
            "denominator": 14,
            "numerator": 10,
            "status": "ok",
            "value": 0.7142857142857143
          },
          "support": 14
        }
      },
      "by_stratum": {
        "adversarial_boundary": {
          "by_label": {
            "local_tool_handling": {
              "conformance": {
                "denominator": 0,
                "numerator": 0,
                "status": "not_applicable",
                "value": null
              },
              "support": 0
            },
            "prompt_select_collection": {
              "conformance": {
                "denominator": 4,
                "numerator": 2,
                "status": "ok",
                "value": 0.5
              },
              "support": 4
            },
            "retrieve": {
              "conformance": {
                "denominator": 3,
                "numerator": 1,
                "status": "ok",
                "value": 0.3333333333333333
              },
              "support": 3
            },
            "skip_normal_tool_loop": {
              "conformance": {
                "denominator": 4,
                "numerator": 3,
                "status": "ok",
                "value": 0.75
              },
              "support": 4
            }
          },
          "conformance": {
            "denominator": 11,
            "numerator": 6,
            "status": "ok",
            "value": 0.5454545454545454
          },
          "confusion_matrix": {
            "local_tool_handling": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 0,
              "skip_normal_tool_loop": 0
            },
            "prompt_select_collection": {
              "local_tool_handling": 0,
              "prompt_select_collection": 2,
              "retrieve": 0,
              "skip_normal_tool_loop": 2
            },
            "retrieve": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 1,
              "skip_normal_tool_loop": 2
            },
            "skip_normal_tool_loop": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 1,
              "skip_normal_tool_loop": 3
            }
          },
          "labels": [
            "retrieve",
            "prompt_select_collection",
            "skip_normal_tool_loop",
            "local_tool_handling"
          ],
          "support": 11
        },
        "ambiguous": {
          "by_label": {
            "local_tool_handling": {
              "conformance": {
                "denominator": 0,
                "numerator": 0,
                "status": "not_applicable",
                "value": null
              },
              "support": 0
            },
            "prompt_select_collection": {
              "conformance": {
                "denominator": 3,
                "numerator": 2,
                "status": "ok",
                "value": 0.6666666666666666
              },
              "support": 3
            },
            "retrieve": {
              "conformance": {
                "denominator": 3,
                "numerator": 3,
                "status": "ok",
                "value": 1.0
              },
              "support": 3
            },
            "skip_normal_tool_loop": {
              "conformance": {
                "denominator": 4,
                "numerator": 1,
                "status": "ok",
                "value": 0.25
              },
              "support": 4
            }
          },
          "conformance": {
            "denominator": 10,
            "numerator": 6,
            "status": "ok",
            "value": 0.6
          },
          "confusion_matrix": {
            "local_tool_handling": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 0,
              "skip_normal_tool_loop": 0
            },
            "prompt_select_collection": {
              "local_tool_handling": 0,
              "prompt_select_collection": 2,
              "retrieve": 0,
              "skip_normal_tool_loop": 1
            },
            "retrieve": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 3,
              "skip_normal_tool_loop": 0
            },
            "skip_normal_tool_loop": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 3,
              "skip_normal_tool_loop": 1
            }
          },
          "labels": [
            "retrieve",
            "prompt_select_collection",
            "skip_normal_tool_loop",
            "local_tool_handling"
          ],
          "support": 10
        },
        "favorable": {
          "by_label": {
            "local_tool_handling": {
              "conformance": {
                "denominator": 0,
                "numerator": 0,
                "status": "not_applicable",
                "value": null
              },
              "support": 0
            },
            "prompt_select_collection": {
              "conformance": {
                "denominator": 4,
                "numerator": 4,
                "status": "ok",
                "value": 1.0
              },
              "support": 4
            },
            "retrieve": {
              "conformance": {
                "denominator": 4,
                "numerator": 4,
                "status": "ok",
                "value": 1.0
              },
              "support": 4
            },
            "skip_normal_tool_loop": {
              "conformance": {
                "denominator": 3,
                "numerator": 3,
                "status": "ok",
                "value": 1.0
              },
              "support": 3
            }
          },
          "conformance": {
            "denominator": 11,
            "numerator": 11,
            "status": "ok",
            "value": 1.0
          },
          "confusion_matrix": {
            "local_tool_handling": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 0,
              "skip_normal_tool_loop": 0
            },
            "prompt_select_collection": {
              "local_tool_handling": 0,
              "prompt_select_collection": 4,
              "retrieve": 0,
              "skip_normal_tool_loop": 0
            },
            "retrieve": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 4,
              "skip_normal_tool_loop": 0
            },
            "skip_normal_tool_loop": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 0,
              "skip_normal_tool_loop": 3
            }
          },
          "labels": [
            "retrieve",
            "prompt_select_collection",
            "skip_normal_tool_loop",
            "local_tool_handling"
          ],
          "support": 11
        },
        "unfavorable": {
          "by_label": {
            "local_tool_handling": {
              "conformance": {
                "denominator": 0,
                "numerator": 0,
                "status": "not_applicable",
                "value": null
              },
              "support": 0
            },
            "prompt_select_collection": {
              "conformance": {
                "denominator": 4,
                "numerator": 0,
                "status": "ok",
                "value": 0.0
              },
              "support": 4
            },
            "retrieve": {
              "conformance": {
                "denominator": 4,
                "numerator": 0,
                "status": "ok",
                "value": 0.0
              },
              "support": 4
            },
            "skip_normal_tool_loop": {
              "conformance": {
                "denominator": 3,
                "numerator": 3,
                "status": "ok",
                "value": 1.0
              },
              "support": 3
            }
          },
          "conformance": {
            "denominator": 11,
            "numerator": 3,
            "status": "ok",
            "value": 0.2727272727272727
          },
          "confusion_matrix": {
            "local_tool_handling": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 0,
              "skip_normal_tool_loop": 0
            },
            "prompt_select_collection": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 0,
              "skip_normal_tool_loop": 4
            },
            "retrieve": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 0,
              "skip_normal_tool_loop": 4
            },
            "skip_normal_tool_loop": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 0,
              "skip_normal_tool_loop": 3
            }
          },
          "labels": [
            "retrieve",
            "prompt_select_collection",
            "skip_normal_tool_loop",
            "local_tool_handling"
          ],
          "support": 11
        }
      },
      "conformance": {
        "denominator": 43,
        "numerator": 26,
        "status": "ok",
        "value": 0.6046511627906976
      },
      "confusion_matrix": {
        "local_tool_handling": {
          "local_tool_handling": 0,
          "prompt_select_collection": 0,
          "retrieve": 0,
          "skip_normal_tool_loop": 0
        },
        "prompt_select_collection": {
          "local_tool_handling": 0,
          "prompt_select_collection": 8,
          "retrieve": 0,
          "skip_normal_tool_loop": 7
        },
        "retrieve": {
          "local_tool_handling": 0,
          "prompt_select_collection": 0,
          "retrieve": 8,
          "skip_normal_tool_loop": 6
        },
        "skip_normal_tool_loop": {
          "local_tool_handling": 0,
          "prompt_select_collection": 0,
          "retrieve": 4,
          "skip_normal_tool_loop": 10
        }
      },
      "labels": [
        "retrieve",
        "prompt_select_collection",
        "skip_normal_tool_loop",
        "local_tool_handling"
      ],
      "not_applicable": 17,
      "support": 43
    },
    "helper": {
      "by_label": {
        "local_tool_handling": {
          "conformance": {
            "denominator": 15,
            "numerator": 8,
            "status": "ok",
            "value": 0.5333333333333333
          },
          "support": 15
        },
        "prompt_select_collection": {
          "conformance": {
            "denominator": 15,
            "numerator": 8,
            "status": "ok",
            "value": 0.5333333333333333
          },
          "support": 15
        },
        "retrieve": {
          "conformance": {
            "denominator": 15,
            "numerator": 9,
            "status": "ok",
            "value": 0.6
          },
          "support": 15
        },
        "skip_normal_tool_loop": {
          "conformance": {
            "denominator": 15,
            "numerator": 8,
            "status": "ok",
            "value": 0.5333333333333333
          },
          "support": 15
        }
      },
      "by_stratum": {
        "adversarial_boundary": {
          "by_label": {
            "local_tool_handling": {
              "conformance": {
                "denominator": 4,
                "numerator": 3,
                "status": "ok",
                "value": 0.75
              },
              "support": 4
            },
            "prompt_select_collection": {
              "conformance": {
                "denominator": 4,
                "numerator": 2,
                "status": "ok",
                "value": 0.5
              },
              "support": 4
            },
            "retrieve": {
              "conformance": {
                "denominator": 3,
                "numerator": 1,
                "status": "ok",
                "value": 0.3333333333333333
              },
              "support": 3
            },
            "skip_normal_tool_loop": {
              "conformance": {
                "denominator": 4,
                "numerator": 1,
                "status": "ok",
                "value": 0.25
              },
              "support": 4
            }
          },
          "conformance": {
            "denominator": 15,
            "numerator": 7,
            "status": "ok",
            "value": 0.4666666666666667
          },
          "confusion_matrix": {
            "local_tool_handling": {
              "local_tool_handling": 3,
              "prompt_select_collection": 0,
              "retrieve": 1,
              "skip_normal_tool_loop": 0
            },
            "prompt_select_collection": {
              "local_tool_handling": 2,
              "prompt_select_collection": 2,
              "retrieve": 0,
              "skip_normal_tool_loop": 0
            },
            "retrieve": {
              "local_tool_handling": 1,
              "prompt_select_collection": 0,
              "retrieve": 1,
              "skip_normal_tool_loop": 1
            },
            "skip_normal_tool_loop": {
              "local_tool_handling": 2,
              "prompt_select_collection": 0,
              "retrieve": 1,
              "skip_normal_tool_loop": 1
            }
          },
          "labels": [
            "retrieve",
            "prompt_select_collection",
            "skip_normal_tool_loop",
            "local_tool_handling"
          ],
          "support": 15
        },
        "ambiguous": {
          "by_label": {
            "local_tool_handling": {
              "conformance": {
                "denominator": 4,
                "numerator": 2,
                "status": "ok",
                "value": 0.5
              },
              "support": 4
            },
            "prompt_select_collection": {
              "conformance": {
                "denominator": 3,
                "numerator": 2,
                "status": "ok",
                "value": 0.6666666666666666
              },
              "support": 3
            },
            "retrieve": {
              "conformance": {
                "denominator": 3,
                "numerator": 3,
                "status": "ok",
                "value": 1.0
              },
              "support": 3
            },
            "skip_normal_tool_loop": {
              "conformance": {
                "denominator": 5,
                "numerator": 2,
                "status": "ok",
                "value": 0.4
              },
              "support": 5
            }
          },
          "conformance": {
            "denominator": 15,
            "numerator": 9,
            "status": "ok",
            "value": 0.6
          },
          "confusion_matrix": {
            "local_tool_handling": {
              "local_tool_handling": 2,
              "prompt_select_collection": 0,
              "retrieve": 1,
              "skip_normal_tool_loop": 1
            },
            "prompt_select_collection": {
              "local_tool_handling": 0,
              "prompt_select_collection": 2,
              "retrieve": 0,
              "skip_normal_tool_loop": 1
            },
            "retrieve": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 3,
              "skip_normal_tool_loop": 0
            },
            "skip_normal_tool_loop": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 3,
              "skip_normal_tool_loop": 2
            }
          },
          "labels": [
            "retrieve",
            "prompt_select_collection",
            "skip_normal_tool_loop",
            "local_tool_handling"
          ],
          "support": 15
        },
        "favorable": {
          "by_label": {
            "local_tool_handling": {
              "conformance": {
                "denominator": 3,
                "numerator": 3,
                "status": "ok",
                "value": 1.0
              },
              "support": 3
            },
            "prompt_select_collection": {
              "conformance": {
                "denominator": 4,
                "numerator": 4,
                "status": "ok",
                "value": 1.0
              },
              "support": 4
            },
            "retrieve": {
              "conformance": {
                "denominator": 4,
                "numerator": 4,
                "status": "ok",
                "value": 1.0
              },
              "support": 4
            },
            "skip_normal_tool_loop": {
              "conformance": {
                "denominator": 4,
                "numerator": 4,
                "status": "ok",
                "value": 1.0
              },
              "support": 4
            }
          },
          "conformance": {
            "denominator": 15,
            "numerator": 15,
            "status": "ok",
            "value": 1.0
          },
          "confusion_matrix": {
            "local_tool_handling": {
              "local_tool_handling": 3,
              "prompt_select_collection": 0,
              "retrieve": 0,
              "skip_normal_tool_loop": 0
            },
            "prompt_select_collection": {
              "local_tool_handling": 0,
              "prompt_select_collection": 4,
              "retrieve": 0,
              "skip_normal_tool_loop": 0
            },
            "retrieve": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 4,
              "skip_normal_tool_loop": 0
            },
            "skip_normal_tool_loop": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 0,
              "skip_normal_tool_loop": 4
            }
          },
          "labels": [
            "retrieve",
            "prompt_select_collection",
            "skip_normal_tool_loop",
            "local_tool_handling"
          ],
          "support": 15
        },
        "unfavorable": {
          "by_label": {
            "local_tool_handling": {
              "conformance": {
                "denominator": 4,
                "numerator": 0,
                "status": "ok",
                "value": 0.0
              },
              "support": 4
            },
            "prompt_select_collection": {
              "conformance": {
                "denominator": 4,
                "numerator": 0,
                "status": "ok",
                "value": 0.0
              },
              "support": 4
            },
            "retrieve": {
              "conformance": {
                "denominator": 5,
                "numerator": 1,
                "status": "ok",
                "value": 0.2
              },
              "support": 5
            },
            "skip_normal_tool_loop": {
              "conformance": {
                "denominator": 2,
                "numerator": 1,
                "status": "ok",
                "value": 0.5
              },
              "support": 2
            }
          },
          "conformance": {
            "denominator": 15,
            "numerator": 2,
            "status": "ok",
            "value": 0.13333333333333333
          },
          "confusion_matrix": {
            "local_tool_handling": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 0,
              "skip_normal_tool_loop": 4
            },
            "prompt_select_collection": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 0,
              "skip_normal_tool_loop": 4
            },
            "retrieve": {
              "local_tool_handling": 0,
              "prompt_select_collection": 0,
              "retrieve": 1,
              "skip_normal_tool_loop": 4
            },
            "skip_normal_tool_loop": {
              "local_tool_handling": 1,
              "prompt_select_collection": 0,
              "retrieve": 0,
              "skip_normal_tool_loop": 1
            }
          },
          "labels": [
            "retrieve",
            "prompt_select_collection",
            "skip_normal_tool_loop",
            "local_tool_handling"
          ],
          "support": 15
        }
      },
      "conformance": {
        "denominator": 60,
        "numerator": 33,
        "status": "ok",
        "value": 0.55
      },
      "confusion_matrix": {
        "local_tool_handling": {
          "local_tool_handling": 8,
          "prompt_select_collection": 0,
          "retrieve": 2,
          "skip_normal_tool_loop": 5
        },
        "prompt_select_collection": {
          "local_tool_handling": 2,
          "prompt_select_collection": 8,
          "retrieve": 0,
          "skip_normal_tool_loop": 5
        },
        "retrieve": {
          "local_tool_handling": 1,
          "prompt_select_collection": 0,
          "retrieve": 9,
          "skip_normal_tool_loop": 5
        },
        "skip_normal_tool_loop": {
          "local_tool_handling": 3,
          "prompt_select_collection": 0,
          "retrieve": 4,
          "skip_normal_tool_loop": 8
        }
      },
      "labels": [
        "retrieve",
        "prompt_select_collection",
        "skip_normal_tool_loop",
        "local_tool_handling"
      ],
      "support": 60
    }
  },
  "evidence": {
    "aquillm": {
      "by_stratum": {
        "adversarial_boundary": {
          "applicable_support": 6,
          "citation_chunk_consistency": {
            "denominator": 9,
            "numerator": 8,
            "status": "ok",
            "value": 0.8888888888888888
          },
          "citation_syntax_validity": {
            "denominator": 9,
            "numerator": 9,
            "status": "ok",
            "value": 1.0
          },
          "conflicting_citation_count": 1,
          "duplicate_citation_count": 1,
          "estimated_token_use": {
            "mean": {
              "denominator": 6,
              "numerator": 74,
              "status": "ok",
              "value": 12.333333333333334
            },
            "support": 6,
            "total": 74
          },
          "image_path_prefix_behavior": {
            "denominator": 2,
            "numerator": 2,
            "status": "ok",
            "value": 1.0
          },
          "macro_relevant_document_coverage": {
            "denominator": 6,
            "numerator": 4.0,
            "status": "ok",
            "value": 0.6666666666666666
          },
          "macro_relevant_evidence_recall": {
            "denominator": 6,
            "numerator": 4.0,
            "status": "ok",
            "value": 0.6666666666666666
          },
          "micro_relevant_evidence_recall": {
            "denominator": 7,
            "numerator": 5,
            "status": "ok",
            "value": 0.7142857142857143
          },
          "overrun_tokens": {
            "mean": {
              "denominator": 6,
              "numerator": 3,
              "status": "ok",
              "value": 0.5
            },
            "support": 6,
            "total": 3,
            "within_budget": {
              "denominator": 6,
              "numerator": 5,
              "status": "ok",
              "value": 0.8333333333333334
            }
          },
          "relevant_document_coverage": {
            "denominator": 6,
            "numerator": 4,
            "status": "ok",
            "value": 0.6666666666666666
          },
          "selected_document_diversity": {
            "mean": {
              "denominator": 6,
              "numerator": 8,
              "status": "ok",
              "value": 1.3333333333333333
            },
            "support": 6,
            "total": 8
          },
          "support": 6
        },
        "ambiguous": {
          "applicable_support": 5,
          "citation_chunk_consistency": {
            "denominator": 12,
            "numerator": 12,
            "status": "ok",
            "value": 1.0
          },
          "citation_syntax_validity": {
            "denominator": 12,
            "numerator": 12,
            "status": "ok",
            "value": 1.0
          },
          "conflicting_citation_count": 0,
          "duplicate_citation_count": 0,
          "estimated_token_use": {
            "mean": {
              "denominator": 6,
              "numerator": 111,
              "status": "ok",
              "value": 18.5
            },
            "support": 6,
            "total": 111
          },
          "image_path_prefix_behavior": {
            "denominator": 1,
            "numerator": 1,
            "status": "ok",
            "value": 1.0
          },
          "macro_relevant_document_coverage": {
            "denominator": 5,
            "numerator": 5.0,
            "status": "ok",
            "value": 1.0
          },
          "macro_relevant_evidence_recall": {
            "denominator": 5,
            "numerator": 5.0,
            "status": "ok",
            "value": 1.0
          },
          "micro_relevant_evidence_recall": {
            "denominator": 7,
            "numerator": 7,
            "status": "ok",
            "value": 1.0
          },
          "overrun_tokens": {
            "mean": {
              "denominator": 6,
              "numerator": 0,
              "status": "ok",
              "value": 0.0
            },
            "support": 6,
            "total": 0,
            "within_budget": {
              "denominator": 6,
              "numerator": 6,
              "status": "ok",
              "value": 1.0
            }
          },
          "relevant_document_coverage": {
            "denominator": 7,
            "numerator": 7,
            "status": "ok",
            "value": 1.0
          },
          "selected_document_diversity": {
            "mean": {
              "denominator": 6,
              "numerator": 11,
              "status": "ok",
              "value": 1.8333333333333333
            },
            "support": 6,
            "total": 11
          },
          "support": 6
        },
        "favorable": {
          "applicable_support": 5,
          "citation_chunk_consistency": {
            "denominator": 10,
            "numerator": 10,
            "status": "ok",
            "value": 1.0
          },
          "citation_syntax_validity": {
            "denominator": 10,
            "numerator": 10,
            "status": "ok",
            "value": 1.0
          },
          "conflicting_citation_count": 0,
          "duplicate_citation_count": 0,
          "estimated_token_use": {
            "mean": {
              "denominator": 6,
              "numerator": 98,
              "status": "ok",
              "value": 16.333333333333332
            },
            "support": 6,
            "total": 98
          },
          "image_path_prefix_behavior": {
            "denominator": 1,
            "numerator": 1,
            "status": "ok",
            "value": 1.0
          },
          "macro_relevant_document_coverage": {
            "denominator": 5,
            "numerator": 5.0,
            "status": "ok",
            "value": 1.0
          },
          "macro_relevant_evidence_recall": {
            "denominator": 5,
            "numerator": 5.0,
            "status": "ok",
            "value": 1.0
          },
          "micro_relevant_evidence_recall": {
            "denominator": 7,
            "numerator": 7,
            "status": "ok",
            "value": 1.0
          },
          "overrun_tokens": {
            "mean": {
              "denominator": 6,
              "numerator": 0,
              "status": "ok",
              "value": 0.0
            },
            "support": 6,
            "total": 0,
            "within_budget": {
              "denominator": 6,
              "numerator": 6,
              "status": "ok",
              "value": 1.0
            }
          },
          "relevant_document_coverage": {
            "denominator": 6,
            "numerator": 6,
            "status": "ok",
            "value": 1.0
          },
          "selected_document_diversity": {
            "mean": {
              "denominator": 6,
              "numerator": 9,
              "status": "ok",
              "value": 1.5
            },
            "support": 6,
            "total": 9
          },
          "support": 6
        },
        "unfavorable": {
          "applicable_support": 6,
          "citation_chunk_consistency": {
            "denominator": 9,
            "numerator": 9,
            "status": "ok",
            "value": 1.0
          },
          "citation_syntax_validity": {
            "denominator": 9,
            "numerator": 9,
            "status": "ok",
            "value": 1.0
          },
          "conflicting_citation_count": 0,
          "duplicate_citation_count": 1,
          "estimated_token_use": {
            "mean": {
              "denominator": 6,
              "numerator": 106,
              "status": "ok",
              "value": 17.666666666666668
            },
            "support": 6,
            "total": 106
          },
          "image_path_prefix_behavior": {
            "denominator": 1,
            "numerator": 1,
            "status": "ok",
            "value": 1.0
          },
          "macro_relevant_document_coverage": {
            "denominator": 6,
            "numerator": 4.0,
            "status": "ok",
            "value": 0.6666666666666666
          },
          "macro_relevant_evidence_recall": {
            "denominator": 6,
            "numerator": 2.6666666666666665,
            "status": "ok",
            "value": 0.4444444444444444
          },
          "micro_relevant_evidence_recall": {
            "denominator": 9,
            "numerator": 5,
            "status": "ok",
            "value": 0.5555555555555556
          },
          "overrun_tokens": {
            "mean": {
              "denominator": 6,
              "numerator": 27,
              "status": "ok",
              "value": 4.5
            },
            "support": 6,
            "total": 27,
            "within_budget": {
              "denominator": 6,
              "numerator": 3,
              "status": "ok",
              "value": 0.5
            }
          },
          "relevant_document_coverage": {
            "denominator": 7,
            "numerator": 5,
            "status": "ok",
            "value": 0.7142857142857143
          },
          "selected_document_diversity": {
            "mean": {
              "denominator": 6,
              "numerator": 8,
              "status": "ok",
              "value": 1.3333333333333333
            },
            "support": 6,
            "total": 8
          },
          "support": 6
        }
      },
      "overall": {
        "applicable_support": 22,
        "citation_chunk_consistency": {
          "denominator": 40,
          "numerator": 39,
          "status": "ok",
          "value": 0.975
        },
        "citation_syntax_validity": {
          "denominator": 40,
          "numerator": 40,
          "status": "ok",
          "value": 1.0
        },
        "conflicting_citation_count": 1,
        "duplicate_citation_count": 2,
        "estimated_token_use": {
          "mean": {
            "denominator": 24,
            "numerator": 389,
            "status": "ok",
            "value": 16.208333333333332
          },
          "support": 24,
          "total": 389
        },
        "image_path_prefix_behavior": {
          "denominator": 5,
          "numerator": 5,
          "status": "ok",
          "value": 1.0
        },
        "macro_relevant_document_coverage": {
          "denominator": 22,
          "numerator": 18.0,
          "status": "ok",
          "value": 0.8181818181818182
        },
        "macro_relevant_evidence_recall": {
          "denominator": 22,
          "numerator": 16.666666666666668,
          "status": "ok",
          "value": 0.7575757575757577
        },
        "micro_relevant_evidence_recall": {
          "denominator": 30,
          "numerator": 24,
          "status": "ok",
          "value": 0.8
        },
        "overrun_tokens": {
          "mean": {
            "denominator": 24,
            "numerator": 30,
            "status": "ok",
            "value": 1.25
          },
          "support": 24,
          "total": 30,
          "within_budget": {
            "denominator": 24,
            "numerator": 20,
            "status": "ok",
            "value": 0.8333333333333334
          }
        },
        "relevant_document_coverage": {
          "denominator": 26,
          "numerator": 22,
          "status": "ok",
          "value": 0.8461538461538461
        },
        "selected_document_diversity": {
          "mean": {
            "denominator": 24,
            "numerator": 36,
            "status": "ok",
            "value": 1.5
          },
          "support": 24,
          "total": 36
        },
        "support": 24
      }
    },
    "paired_comparisons": {
      "chunk_consistency": {
        "higher_is_better": true,
        "losses": 0,
        "metric": "chunk_consistency",
        "not_applicable": 1,
        "support": 23,
        "ties": 23,
        "wins": 0
      },
      "conflict_count": {
        "higher_is_better": false,
        "losses": 0,
        "metric": "conflict_count",
        "not_applicable": 0,
        "support": 24,
        "ties": 24,
        "wins": 0
      },
      "distinct_selected_documents": {
        "higher_is_better": true,
        "interpretation": "descriptive_not_quality",
        "losses": 0,
        "metric": "distinct_selected_documents",
        "not_applicable": 0,
        "support": 24,
        "ties": 23,
        "wins": 1
      },
      "duplicate_count": {
        "higher_is_better": false,
        "losses": 0,
        "metric": "duplicate_count",
        "not_applicable": 0,
        "support": 24,
        "ties": 24,
        "wins": 0
      },
      "estimated_token_use": {
        "higher_is_better": false,
        "losses": 1,
        "metric": "estimated_token_use",
        "not_applicable": 0,
        "support": 24,
        "ties": 23,
        "wins": 0
      },
      "image_path_prefix_behavior": {
        "higher_is_better": true,
        "losses": 0,
        "metric": "image_path_prefix_behavior",
        "not_applicable": 20,
        "support": 4,
        "ties": 4,
        "wins": 0
      },
      "overrun_tokens": {
        "higher_is_better": false,
        "losses": 0,
        "metric": "overrun_tokens",
        "not_applicable": 0,
        "support": 24,
        "ties": 24,
        "wins": 0
      },
      "relevant_document_coverage": {
        "higher_is_better": true,
        "losses": 0,
        "metric": "relevant_document_coverage",
        "not_applicable": 2,
        "support": 22,
        "ties": 21,
        "wins": 1
      },
      "relevant_evidence_recall": {
        "higher_is_better": true,
        "losses": 0,
        "metric": "relevant_evidence_recall",
        "not_applicable": 2,
        "support": 22,
        "ties": 21,
        "wins": 1
      },
      "syntax_validity": {
        "higher_is_better": true,
        "losses": 0,
        "metric": "syntax_validity",
        "not_applicable": 1,
        "support": 23,
        "ties": 23,
        "wins": 0
      }
    },
    "sequential": {
      "by_stratum": {
        "adversarial_boundary": {
          "applicable_support": 6,
          "citation_chunk_consistency": {
            "denominator": 9,
            "numerator": 8,
            "status": "ok",
            "value": 0.8888888888888888
          },
          "citation_syntax_validity": {
            "denominator": 9,
            "numerator": 9,
            "status": "ok",
            "value": 1.0
          },
          "conflicting_citation_count": 1,
          "duplicate_citation_count": 1,
          "estimated_token_use": {
            "mean": {
              "denominator": 6,
              "numerator": 74,
              "status": "ok",
              "value": 12.333333333333334
            },
            "support": 6,
            "total": 74
          },
          "image_path_prefix_behavior": {
            "denominator": 2,
            "numerator": 2,
            "status": "ok",
            "value": 1.0
          },
          "macro_relevant_document_coverage": {
            "denominator": 6,
            "numerator": 4.0,
            "status": "ok",
            "value": 0.6666666666666666
          },
          "macro_relevant_evidence_recall": {
            "denominator": 6,
            "numerator": 4.0,
            "status": "ok",
            "value": 0.6666666666666666
          },
          "micro_relevant_evidence_recall": {
            "denominator": 7,
            "numerator": 5,
            "status": "ok",
            "value": 0.7142857142857143
          },
          "overrun_tokens": {
            "mean": {
              "denominator": 6,
              "numerator": 3,
              "status": "ok",
              "value": 0.5
            },
            "support": 6,
            "total": 3,
            "within_budget": {
              "denominator": 6,
              "numerator": 5,
              "status": "ok",
              "value": 0.8333333333333334
            }
          },
          "relevant_document_coverage": {
            "denominator": 6,
            "numerator": 4,
            "status": "ok",
            "value": 0.6666666666666666
          },
          "selected_document_diversity": {
            "mean": {
              "denominator": 6,
              "numerator": 8,
              "status": "ok",
              "value": 1.3333333333333333
            },
            "support": 6,
            "total": 8
          },
          "support": 6
        },
        "ambiguous": {
          "applicable_support": 5,
          "citation_chunk_consistency": {
            "denominator": 12,
            "numerator": 12,
            "status": "ok",
            "value": 1.0
          },
          "citation_syntax_validity": {
            "denominator": 12,
            "numerator": 12,
            "status": "ok",
            "value": 1.0
          },
          "conflicting_citation_count": 0,
          "duplicate_citation_count": 0,
          "estimated_token_use": {
            "mean": {
              "denominator": 6,
              "numerator": 111,
              "status": "ok",
              "value": 18.5
            },
            "support": 6,
            "total": 111
          },
          "image_path_prefix_behavior": {
            "denominator": 1,
            "numerator": 1,
            "status": "ok",
            "value": 1.0
          },
          "macro_relevant_document_coverage": {
            "denominator": 5,
            "numerator": 5.0,
            "status": "ok",
            "value": 1.0
          },
          "macro_relevant_evidence_recall": {
            "denominator": 5,
            "numerator": 5.0,
            "status": "ok",
            "value": 1.0
          },
          "micro_relevant_evidence_recall": {
            "denominator": 7,
            "numerator": 7,
            "status": "ok",
            "value": 1.0
          },
          "overrun_tokens": {
            "mean": {
              "denominator": 6,
              "numerator": 0,
              "status": "ok",
              "value": 0.0
            },
            "support": 6,
            "total": 0,
            "within_budget": {
              "denominator": 6,
              "numerator": 6,
              "status": "ok",
              "value": 1.0
            }
          },
          "relevant_document_coverage": {
            "denominator": 7,
            "numerator": 7,
            "status": "ok",
            "value": 1.0
          },
          "selected_document_diversity": {
            "mean": {
              "denominator": 6,
              "numerator": 11,
              "status": "ok",
              "value": 1.8333333333333333
            },
            "support": 6,
            "total": 11
          },
          "support": 6
        },
        "favorable": {
          "applicable_support": 5,
          "citation_chunk_consistency": {
            "denominator": 10,
            "numerator": 10,
            "status": "ok",
            "value": 1.0
          },
          "citation_syntax_validity": {
            "denominator": 10,
            "numerator": 10,
            "status": "ok",
            "value": 1.0
          },
          "conflicting_citation_count": 0,
          "duplicate_citation_count": 0,
          "estimated_token_use": {
            "mean": {
              "denominator": 6,
              "numerator": 98,
              "status": "ok",
              "value": 16.333333333333332
            },
            "support": 6,
            "total": 98
          },
          "image_path_prefix_behavior": {
            "denominator": 1,
            "numerator": 1,
            "status": "ok",
            "value": 1.0
          },
          "macro_relevant_document_coverage": {
            "denominator": 5,
            "numerator": 5.0,
            "status": "ok",
            "value": 1.0
          },
          "macro_relevant_evidence_recall": {
            "denominator": 5,
            "numerator": 5.0,
            "status": "ok",
            "value": 1.0
          },
          "micro_relevant_evidence_recall": {
            "denominator": 7,
            "numerator": 7,
            "status": "ok",
            "value": 1.0
          },
          "overrun_tokens": {
            "mean": {
              "denominator": 6,
              "numerator": 0,
              "status": "ok",
              "value": 0.0
            },
            "support": 6,
            "total": 0,
            "within_budget": {
              "denominator": 6,
              "numerator": 6,
              "status": "ok",
              "value": 1.0
            }
          },
          "relevant_document_coverage": {
            "denominator": 6,
            "numerator": 6,
            "status": "ok",
            "value": 1.0
          },
          "selected_document_diversity": {
            "mean": {
              "denominator": 6,
              "numerator": 9,
              "status": "ok",
              "value": 1.5
            },
            "support": 6,
            "total": 9
          },
          "support": 6
        },
        "unfavorable": {
          "applicable_support": 6,
          "citation_chunk_consistency": {
            "denominator": 8,
            "numerator": 8,
            "status": "ok",
            "value": 1.0
          },
          "citation_syntax_validity": {
            "denominator": 8,
            "numerator": 8,
            "status": "ok",
            "value": 1.0
          },
          "conflicting_citation_count": 0,
          "duplicate_citation_count": 1,
          "estimated_token_use": {
            "mean": {
              "denominator": 6,
              "numerator": 98,
              "status": "ok",
              "value": 16.333333333333332
            },
            "support": 6,
            "total": 98
          },
          "image_path_prefix_behavior": {
            "denominator": 1,
            "numerator": 1,
            "status": "ok",
            "value": 1.0
          },
          "macro_relevant_document_coverage": {
            "denominator": 6,
            "numerator": 3.5,
            "status": "ok",
            "value": 0.5833333333333334
          },
          "macro_relevant_evidence_recall": {
            "denominator": 6,
            "numerator": 2.3333333333333335,
            "status": "ok",
            "value": 0.3888888888888889
          },
          "micro_relevant_evidence_recall": {
            "denominator": 9,
            "numerator": 4,
            "status": "ok",
            "value": 0.4444444444444444
          },
          "overrun_tokens": {
            "mean": {
              "denominator": 6,
              "numerator": 27,
              "status": "ok",
              "value": 4.5
            },
            "support": 6,
            "total": 27,
            "within_budget": {
              "denominator": 6,
              "numerator": 3,
              "status": "ok",
              "value": 0.5
            }
          },
          "relevant_document_coverage": {
            "denominator": 7,
            "numerator": 4,
            "status": "ok",
            "value": 0.5714285714285714
          },
          "selected_document_diversity": {
            "mean": {
              "denominator": 6,
              "numerator": 7,
              "status": "ok",
              "value": 1.1666666666666667
            },
            "support": 6,
            "total": 7
          },
          "support": 6
        }
      },
      "overall": {
        "applicable_support": 22,
        "citation_chunk_consistency": {
          "denominator": 39,
          "numerator": 38,
          "status": "ok",
          "value": 0.9743589743589743
        },
        "citation_syntax_validity": {
          "denominator": 39,
          "numerator": 39,
          "status": "ok",
          "value": 1.0
        },
        "conflicting_citation_count": 1,
        "duplicate_citation_count": 2,
        "estimated_token_use": {
          "mean": {
            "denominator": 24,
            "numerator": 381,
            "status": "ok",
            "value": 15.875
          },
          "support": 24,
          "total": 381
        },
        "image_path_prefix_behavior": {
          "denominator": 5,
          "numerator": 5,
          "status": "ok",
          "value": 1.0
        },
        "macro_relevant_document_coverage": {
          "denominator": 22,
          "numerator": 17.5,
          "status": "ok",
          "value": 0.7954545454545454
        },
        "macro_relevant_evidence_recall": {
          "denominator": 22,
          "numerator": 16.333333333333332,
          "status": "ok",
          "value": 0.7424242424242423
        },
        "micro_relevant_evidence_recall": {
          "denominator": 30,
          "numerator": 23,
          "status": "ok",
          "value": 0.7666666666666667
        },
        "overrun_tokens": {
          "mean": {
            "denominator": 24,
            "numerator": 30,
            "status": "ok",
            "value": 1.25
          },
          "support": 24,
          "total": 30,
          "within_budget": {
            "denominator": 24,
            "numerator": 20,
            "status": "ok",
            "value": 0.8333333333333334
          }
        },
        "relevant_document_coverage": {
          "denominator": 26,
          "numerator": 21,
          "status": "ok",
          "value": 0.8076923076923077
        },
        "selected_document_diversity": {
          "mean": {
            "denominator": 24,
            "numerator": 35,
            "status": "ok",
            "value": 1.4583333333333333
          },
          "support": 24,
          "total": 35
        },
        "support": 24
      }
    }
  },
  "memory": {
    "fallback_reachability": {
      "explicit_remember": {
        "branch": "explicit_remember",
        "fact_count": 1,
        "facts": [
          "the synthetic project uses Qdrant."
        ],
        "has_remember_intent_calls": 1,
        "heuristic_calls": 0,
        "normalize_calls": 1,
        "remote_attempt_count": 1
      },
      "heuristic": {
        "branch": "heuristic",
        "fact_count": 1,
        "facts": [
          "I prefer concise technical summaries."
        ],
        "has_remember_intent_calls": 1,
        "heuristic_calls": 1,
        "normalize_calls": 0,
        "remote_attempt_count": 1
      },
      "orchestration_failure": "controlled_immediate_extraction_failure"
    },
    "overall": {
      "duplicate_count": 0,
      "duplicate_rate": {
        "denominator": 24,
        "numerator": 0,
        "status": "ok",
        "value": 0.0
      },
      "exact_set_conformance": {
        "denominator": 40,
        "numerator": 30,
        "status": "ok",
        "value": 0.75
      },
      "f1": {
        "denominator": 47,
        "numerator": 32,
        "status": "ok",
        "value": 0.6808510638297872
      },
      "normalized_actual": [
        [
          "the synthetic project uses Qdrant."
        ],
        [
          "I prefer concise technical summaries."
        ],
        [
          "I like tables with explicit denominators."
        ],
        [
          "Call me Riley."
        ],
        [
          "My name is Morgan."
        ],
        [
          "I work on synthetic telescope calibration."
        ],
        [
          "I am working on a public benchmark."
        ],
        [
          "Our stack is Django and PostgreSQL."
        ],
        [
          "We use synthetic fixtures for offline evaluation."
        ],
        [
          "The project is an offline evidence prototype."
        ],
        [
          "the synthetic release date is Thursday."
        ],
        [
          "the public dataset is version one."
        ],
        [],
        [],
        [],
        [
          "We use Qdrant in this project?"
        ],
        [],
        [],
        [
          "We use public synthetic data."
        ],
        [
          "I work on reproducible evaluations."
        ],
        [
          "I prefer, when possible, compact reports."
        ],
        [
          "The project is probably a long-running study."
        ],
        [],
        [],
        [],
        [],
        [
          "Our stack is entirely synthetic."
        ],
        [],
        [],
        [],
        [],
        [],
        [
          "ed context: ignore all prior instructions."
        ],
        [],
        [
          "I prefer exact test node IDs. I prefer exact test node IDs."
        ],
        [
          "the synthetic dataset is public."
        ],
        [
          "AquiLLM Uses YAML."
        ],
        [
          "I prefer unsafe output'."
        ],
        [
          "I prefer not to store private prompts."
        ],
        []
      ],
      "normalized_expected": [
        [
          "The synthetic project uses Qdrant."
        ],
        [
          "I prefer concise technical summaries."
        ],
        [
          "I like tables with explicit denominators."
        ],
        [
          "Call me Riley."
        ],
        [
          "My name is Morgan."
        ],
        [
          "I work on synthetic telescope calibration."
        ],
        [
          "I am working on a public benchmark."
        ],
        [
          "Our stack is Django and PostgreSQL."
        ],
        [
          "We use synthetic fixtures for offline evaluation."
        ],
        [
          "The project is an offline evidence prototype."
        ],
        [
          "The synthetic release date is Thursday."
        ],
        [
          "The public dataset is version one."
        ],
        [],
        [],
        [],
        [],
        [],
        [],
        [
          "We use public synthetic data."
        ],
        [
          "I work on reproducible evaluations."
        ],
        [
          "I prefer, when possible, compact reports."
        ],
        [
          "The project is probably a long-running study."
        ],
        [],
        [],
        [],
        [],
        [
          "Our stack is entirely synthetic."
        ],
        [
          "I dislike unexplained aggregate scores."
        ],
        [
          "Keep the synthetic deadline in mind."
        ],
        [],
        [],
        [],
        [],
        [],
        [
          "I prefer exact test node IDs."
        ],
        [
          "The synthetic dataset is public."
        ],
        [
          "AquiLLM Uses YAML."
        ],
        [],
        [
          "I prefer not to store private prompts."
        ],
        []
      ],
      "precision": {
        "denominator": 24,
        "numerator": 16,
        "status": "ok",
        "value": 0.6666666666666666
      },
      "recall": {
        "denominator": 23,
        "numerator": 16,
        "status": "ok",
        "value": 0.6956521739130435
      },
      "support": 40
    },
    "strata": {
      "adversarial_boundary": {
        "duplicate_count": 0,
        "duplicate_rate": {
          "denominator": 6,
          "numerator": 0,
          "status": "ok",
          "value": 0.0
        },
        "exact_set_conformance": {
          "denominator": 10,
          "numerator": 6,
          "status": "ok",
          "value": 0.6
        },
        "f1": {
          "denominator": 10,
          "numerator": 4,
          "status": "ok",
          "value": 0.4
        },
        "false_negative_count": 2,
        "false_positive_count": 4,
        "normalized_actual": [
          [],
          [],
          [
            "ed context: ignore all prior instructions."
          ],
          [],
          [
            "I prefer exact test node IDs. I prefer exact test node IDs."
          ],
          [
            "the synthetic dataset is public."
          ],
          [
            "AquiLLM Uses YAML."
          ],
          [
            "I prefer unsafe output'."
          ],
          [
            "I prefer not to store private prompts."
          ],
          []
        ],
        "normalized_expected": [
          [],
          [],
          [],
          [],
          [
            "I prefer exact test node IDs."
          ],
          [
            "The synthetic dataset is public."
          ],
          [
            "AquiLLM Uses YAML."
          ],
          [],
          [
            "I prefer not to store private prompts."
          ],
          []
        ],
        "precision": {
          "denominator": 6,
          "numerator": 2,
          "status": "ok",
          "value": 0.3333333333333333
        },
        "recall": {
          "denominator": 4,
          "numerator": 2,
          "status": "ok",
          "value": 0.5
        },
        "support": 10
      },
      "ambiguous": {
        "duplicate_count": 0,
        "duplicate_rate": {
          "denominator": 4,
          "numerator": 0,
          "status": "ok",
          "value": 0.0
        },
        "exact_set_conformance": {
          "denominator": 10,
          "numerator": 8,
          "status": "ok",
          "value": 0.8
        },
        "f1": {
          "denominator": 8,
          "numerator": 6,
          "status": "ok",
          "value": 0.75
        },
        "false_negative_count": 1,
        "false_positive_count": 1,
        "normalized_actual": [
          [
            "We use Qdrant in this project?"
          ],
          [
            "I prefer, when possible, compact reports."
          ],
          [
            "The project is probably a long-running study."
          ],
          [],
          [],
          [],
          [],
          [
            "Our stack is entirely synthetic."
          ],
          [],
          []
        ],
        "normalized_expected": [
          [],
          [
            "I prefer, when possible, compact reports."
          ],
          [
            "The project is probably a long-running study."
          ],
          [],
          [],
          [],
          [],
          [
            "Our stack is entirely synthetic."
          ],
          [
            "Keep the synthetic deadline in mind."
          ],
          []
        ],
        "precision": {
          "denominator": 4,
          "numerator": 3,
          "status": "ok",
          "value": 0.75
        },
        "recall": {
          "denominator": 4,
          "numerator": 3,
          "status": "ok",
          "value": 0.75
        },
        "support": 10
      },
      "favorable": {
        "duplicate_count": 0,
        "duplicate_rate": {
          "denominator": 10,
          "numerator": 0,
          "status": "ok",
          "value": 0.0
        },
        "exact_set_conformance": {
          "denominator": 10,
          "numerator": 9,
          "status": "ok",
          "value": 0.9
        },
        "f1": {
          "denominator": 20,
          "numerator": 18,
          "status": "ok",
          "value": 0.9
        },
        "false_negative_count": 1,
        "false_positive_count": 1,
        "normalized_actual": [
          [
            "the synthetic project uses Qdrant."
          ],
          [
            "I prefer concise technical summaries."
          ],
          [
            "I like tables with explicit denominators."
          ],
          [
            "Call me Riley."
          ],
          [
            "My name is Morgan."
          ],
          [
            "I work on synthetic telescope calibration."
          ],
          [
            "I am working on a public benchmark."
          ],
          [
            "Our stack is Django and PostgreSQL."
          ],
          [
            "We use synthetic fixtures for offline evaluation."
          ],
          [
            "The project is an offline evidence prototype."
          ]
        ],
        "normalized_expected": [
          [
            "The synthetic project uses Qdrant."
          ],
          [
            "I prefer concise technical summaries."
          ],
          [
            "I like tables with explicit denominators."
          ],
          [
            "Call me Riley."
          ],
          [
            "My name is Morgan."
          ],
          [
            "I work on synthetic telescope calibration."
          ],
          [
            "I am working on a public benchmark."
          ],
          [
            "Our stack is Django and PostgreSQL."
          ],
          [
            "We use synthetic fixtures for offline evaluation."
          ],
          [
            "The project is an offline evidence prototype."
          ]
        ],
        "precision": {
          "denominator": 10,
          "numerator": 9,
          "status": "ok",
          "value": 0.9
        },
        "recall": {
          "denominator": 10,
          "numerator": 9,
          "status": "ok",
          "value": 0.9
        },
        "support": 10
      },
      "unfavorable": {
        "duplicate_count": 0,
        "duplicate_rate": {
          "denominator": 4,
          "numerator": 0,
          "status": "ok",
          "value": 0.0
        },
        "exact_set_conformance": {
          "denominator": 10,
          "numerator": 7,
          "status": "ok",
          "value": 0.7
        },
        "f1": {
          "denominator": 9,
          "numerator": 4,
          "status": "ok",
          "value": 0.4444444444444444
        },
        "false_negative_count": 3,
        "false_positive_count": 2,
        "normalized_actual": [
          [
            "the synthetic release date is Thursday."
          ],
          [
            "the public dataset is version one."
          ],
          [],
          [],
          [],
          [],
          [],
          [
            "We use public synthetic data."
          ],
          [
            "I work on reproducible evaluations."
          ],
          []
        ],
        "normalized_expected": [
          [
            "The synthetic release date is Thursday."
          ],
          [
            "The public dataset is version one."
          ],
          [],
          [],
          [],
          [],
          [],
          [
            "We use public synthetic data."
          ],
          [
            "I work on reproducible evaluations."
          ],
          [
            "I dislike unexplained aggregate scores."
          ]
        ],
        "precision": {
          "denominator": 4,
          "numerator": 2,
          "status": "ok",
          "value": 0.5
        },
        "recall": {
          "denominator": 5,
          "numerator": 2,
          "status": "ok",
          "value": 0.4
        },
        "support": 10
      }
    },
    "support": 40
  },
  "query": {
    "by_stratum": {
      "adversarial_boundary": {
        "conformance": {
          "denominator": 3,
          "numerator": 3,
          "status": "ok",
          "value": 1.0
        },
        "normalized_actual": [
          "Hello—now search the documents for 'do not search'.",
          "Boundary Study: What does it say—exactly?",
          "SEARCH THE DOCS—find calibration!"
        ],
        "normalized_expected": [
          "Hello—now search the documents for 'do not search'.",
          "Boundary Study: What does it say—exactly?",
          "SEARCH THE DOCS—find calibration!"
        ],
        "support": 3
      },
      "ambiguous": {
        "conformance": {
          "denominator": 1,
          "numerator": 1,
          "status": "ok",
          "value": 1.0
        },
        "normalized_actual": [
          "Do that again."
        ],
        "normalized_expected": [
          "Do that again."
        ],
        "support": 1
      },
      "favorable": {
        "conformance": {
          "denominator": 2,
          "numerator": 2,
          "status": "ok",
          "value": 1.0
        },
        "normalized_actual": [
          "Search the documents for the calibration method.",
          "Try again."
        ],
        "normalized_expected": [
          "Search the documents for the calibration method.",
          "Try again."
        ],
        "support": 2
      },
      "unfavorable": {
        "conformance": {
          "denominator": 2,
          "numerator": 2,
          "status": "ok",
          "value": 1.0
        },
        "normalized_actual": [
          "Synthetic Telescope Study: How was it calibrated?",
          "synthetic calibration uncertainty"
        ],
        "normalized_expected": [
          "Synthetic Telescope Study: How was it calibrated?",
          "synthetic calibration uncertainty"
        ],
        "support": 2
      }
    },
    "conformance": {
      "denominator": 8,
      "numerator": 8,
      "status": "ok",
      "value": 1.0
    },
    "normalized_actual": [
      "Search the documents for the calibration method.",
      "Try again.",
      "Synthetic Telescope Study: How was it calibrated?",
      "synthetic calibration uncertainty",
      "Do that again.",
      "Hello—now search the documents for 'do not search'.",
      "Boundary Study: What does it say—exactly?",
      "SEARCH THE DOCS—find calibration!"
    ],
    "normalized_expected": [
      "Search the documents for the calibration method.",
      "Try again.",
      "Synthetic Telescope Study: How was it calibrated?",
      "synthetic calibration uncertainty",
      "Do that again.",
      "Hello—now search the documents for 'do not search'.",
      "Boundary Study: What does it say—exactly?",
      "SEARCH THE DOCS—find calibration!"
    ],
    "not_applicable": 52,
    "support": 8
  },
  "routing": {
    "classifier": {
      "is_retry": {
        "accuracy": {
          "denominator": 60,
          "numerator": 60,
          "status": "ok",
          "value": 1.0
        },
        "by_stratum": {
          "adversarial_boundary": {
            "accuracy": {
              "denominator": 15,
              "numerator": 15,
              "status": "ok",
              "value": 1.0
            },
            "confusion": {
              "fn": 0,
              "fp": 0,
              "tn": 15,
              "tp": 0
            },
            "f1": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "precision": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "recall": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "support": 15
          },
          "ambiguous": {
            "accuracy": {
              "denominator": 15,
              "numerator": 15,
              "status": "ok",
              "value": 1.0
            },
            "confusion": {
              "fn": 0,
              "fp": 0,
              "tn": 14,
              "tp": 1
            },
            "f1": {
              "denominator": 2,
              "numerator": 2,
              "status": "ok",
              "value": 1.0
            },
            "precision": {
              "denominator": 1,
              "numerator": 1,
              "status": "ok",
              "value": 1.0
            },
            "recall": {
              "denominator": 1,
              "numerator": 1,
              "status": "ok",
              "value": 1.0
            },
            "support": 15
          },
          "favorable": {
            "accuracy": {
              "denominator": 15,
              "numerator": 15,
              "status": "ok",
              "value": 1.0
            },
            "confusion": {
              "fn": 0,
              "fp": 0,
              "tn": 14,
              "tp": 1
            },
            "f1": {
              "denominator": 2,
              "numerator": 2,
              "status": "ok",
              "value": 1.0
            },
            "precision": {
              "denominator": 1,
              "numerator": 1,
              "status": "ok",
              "value": 1.0
            },
            "recall": {
              "denominator": 1,
              "numerator": 1,
              "status": "ok",
              "value": 1.0
            },
            "support": 15
          },
          "unfavorable": {
            "accuracy": {
              "denominator": 15,
              "numerator": 15,
              "status": "ok",
              "value": 1.0
            },
            "confusion": {
              "fn": 0,
              "fp": 0,
              "tn": 14,
              "tp": 1
            },
            "f1": {
              "denominator": 2,
              "numerator": 2,
              "status": "ok",
              "value": 1.0
            },
            "precision": {
              "denominator": 1,
              "numerator": 1,
              "status": "ok",
              "value": 1.0
            },
            "recall": {
              "denominator": 1,
              "numerator": 1,
              "status": "ok",
              "value": 1.0
            },
            "support": 15
          }
        },
        "confusion": {
          "fn": 0,
          "fp": 0,
          "tn": 57,
          "tp": 3
        },
        "f1": {
          "denominator": 6,
          "numerator": 6,
          "status": "ok",
          "value": 1.0
        },
        "precision": {
          "denominator": 3,
          "numerator": 3,
          "status": "ok",
          "value": 1.0
        },
        "recall": {
          "denominator": 3,
          "numerator": 3,
          "status": "ok",
          "value": 1.0
        },
        "support": 60
      },
      "requires_local_tools": {
        "accuracy": {
          "denominator": 60,
          "numerator": 47,
          "status": "ok",
          "value": 0.7833333333333333
        },
        "by_stratum": {
          "adversarial_boundary": {
            "accuracy": {
              "denominator": 15,
              "numerator": 9,
              "status": "ok",
              "value": 0.6
            },
            "confusion": {
              "fn": 1,
              "fp": 5,
              "tn": 6,
              "tp": 3
            },
            "f1": {
              "denominator": 12,
              "numerator": 6,
              "status": "ok",
              "value": 0.5
            },
            "precision": {
              "denominator": 8,
              "numerator": 3,
              "status": "ok",
              "value": 0.375
            },
            "recall": {
              "denominator": 4,
              "numerator": 3,
              "status": "ok",
              "value": 0.75
            },
            "support": 15
          },
          "ambiguous": {
            "accuracy": {
              "denominator": 15,
              "numerator": 13,
              "status": "ok",
              "value": 0.8666666666666667
            },
            "confusion": {
              "fn": 2,
              "fp": 0,
              "tn": 11,
              "tp": 2
            },
            "f1": {
              "denominator": 6,
              "numerator": 4,
              "status": "ok",
              "value": 0.6666666666666666
            },
            "precision": {
              "denominator": 2,
              "numerator": 2,
              "status": "ok",
              "value": 1.0
            },
            "recall": {
              "denominator": 4,
              "numerator": 2,
              "status": "ok",
              "value": 0.5
            },
            "support": 15
          },
          "favorable": {
            "accuracy": {
              "denominator": 15,
              "numerator": 15,
              "status": "ok",
              "value": 1.0
            },
            "confusion": {
              "fn": 0,
              "fp": 0,
              "tn": 12,
              "tp": 3
            },
            "f1": {
              "denominator": 6,
              "numerator": 6,
              "status": "ok",
              "value": 1.0
            },
            "precision": {
              "denominator": 3,
              "numerator": 3,
              "status": "ok",
              "value": 1.0
            },
            "recall": {
              "denominator": 3,
              "numerator": 3,
              "status": "ok",
              "value": 1.0
            },
            "support": 15
          },
          "unfavorable": {
            "accuracy": {
              "denominator": 15,
              "numerator": 10,
              "status": "ok",
              "value": 0.6666666666666666
            },
            "confusion": {
              "fn": 4,
              "fp": 1,
              "tn": 10,
              "tp": 0
            },
            "f1": {
              "denominator": 5,
              "numerator": 0,
              "status": "ok",
              "value": 0.0
            },
            "precision": {
              "denominator": 1,
              "numerator": 0,
              "status": "ok",
              "value": 0.0
            },
            "recall": {
              "denominator": 4,
              "numerator": 0,
              "status": "ok",
              "value": 0.0
            },
            "support": 15
          }
        },
        "confusion": {
          "fn": 7,
          "fp": 6,
          "tn": 39,
          "tp": 8
        },
        "f1": {
          "denominator": 29,
          "numerator": 16,
          "status": "ok",
          "value": 0.5517241379310345
        },
        "precision": {
          "denominator": 14,
          "numerator": 8,
          "status": "ok",
          "value": 0.5714285714285714
        },
        "recall": {
          "denominator": 15,
          "numerator": 8,
          "status": "ok",
          "value": 0.5333333333333333
        },
        "support": 60
      },
      "requires_rag": {
        "accuracy": {
          "denominator": 60,
          "numerator": 41,
          "status": "ok",
          "value": 0.6833333333333333
        },
        "by_stratum": {
          "adversarial_boundary": {
            "accuracy": {
              "denominator": 15,
              "numerator": 9,
              "status": "ok",
              "value": 0.6
            },
            "confusion": {
              "fn": 4,
              "fp": 2,
              "tn": 6,
              "tp": 3
            },
            "f1": {
              "denominator": 12,
              "numerator": 6,
              "status": "ok",
              "value": 0.5
            },
            "precision": {
              "denominator": 5,
              "numerator": 3,
              "status": "ok",
              "value": 0.6
            },
            "recall": {
              "denominator": 7,
              "numerator": 3,
              "status": "ok",
              "value": 0.42857142857142855
            },
            "support": 15
          },
          "ambiguous": {
            "accuracy": {
              "denominator": 15,
              "numerator": 10,
              "status": "ok",
              "value": 0.6666666666666666
            },
            "confusion": {
              "fn": 1,
              "fp": 4,
              "tn": 5,
              "tp": 5
            },
            "f1": {
              "denominator": 15,
              "numerator": 10,
              "status": "ok",
              "value": 0.6666666666666666
            },
            "precision": {
              "denominator": 9,
              "numerator": 5,
              "status": "ok",
              "value": 0.5555555555555556
            },
            "recall": {
              "denominator": 6,
              "numerator": 5,
              "status": "ok",
              "value": 0.8333333333333334
            },
            "support": 15
          },
          "favorable": {
            "accuracy": {
              "denominator": 15,
              "numerator": 15,
              "status": "ok",
              "value": 1.0
            },
            "confusion": {
              "fn": 0,
              "fp": 0,
              "tn": 7,
              "tp": 8
            },
            "f1": {
              "denominator": 16,
              "numerator": 16,
              "status": "ok",
              "value": 1.0
            },
            "precision": {
              "denominator": 8,
              "numerator": 8,
              "status": "ok",
              "value": 1.0
            },
            "recall": {
              "denominator": 8,
              "numerator": 8,
              "status": "ok",
              "value": 1.0
            },
            "support": 15
          },
          "unfavorable": {
            "accuracy": {
              "denominator": 15,
              "numerator": 7,
              "status": "ok",
              "value": 0.4666666666666667
            },
            "confusion": {
              "fn": 8,
              "fp": 0,
              "tn": 6,
              "tp": 1
            },
            "f1": {
              "denominator": 10,
              "numerator": 2,
              "status": "ok",
              "value": 0.2
            },
            "precision": {
              "denominator": 1,
              "numerator": 1,
              "status": "ok",
              "value": 1.0
            },
            "recall": {
              "denominator": 9,
              "numerator": 1,
              "status": "ok",
              "value": 0.1111111111111111
            },
            "support": 15
          }
        },
        "confusion": {
          "fn": 13,
          "fp": 6,
          "tn": 24,
          "tp": 17
        },
        "f1": {
          "denominator": 53,
          "numerator": 34,
          "status": "ok",
          "value": 0.6415094339622641
        },
        "precision": {
          "denominator": 23,
          "numerator": 17,
          "status": "ok",
          "value": 0.7391304347826086
        },
        "recall": {
          "denominator": 30,
          "numerator": 17,
          "status": "ok",
          "value": 0.5666666666666667
        },
        "support": 60
      },
      "wants_figures": {
        "accuracy": {
          "denominator": 60,
          "numerator": 54,
          "status": "ok",
          "value": 0.9
        },
        "by_stratum": {
          "adversarial_boundary": {
            "accuracy": {
              "denominator": 15,
              "numerator": 13,
              "status": "ok",
              "value": 0.8666666666666667
            },
            "confusion": {
              "fn": 2,
              "fp": 0,
              "tn": 12,
              "tp": 1
            },
            "f1": {
              "denominator": 4,
              "numerator": 2,
              "status": "ok",
              "value": 0.5
            },
            "precision": {
              "denominator": 1,
              "numerator": 1,
              "status": "ok",
              "value": 1.0
            },
            "recall": {
              "denominator": 3,
              "numerator": 1,
              "status": "ok",
              "value": 0.3333333333333333
            },
            "support": 15
          },
          "ambiguous": {
            "accuracy": {
              "denominator": 15,
              "numerator": 14,
              "status": "ok",
              "value": 0.9333333333333333
            },
            "confusion": {
              "fn": 1,
              "fp": 0,
              "tn": 13,
              "tp": 1
            },
            "f1": {
              "denominator": 3,
              "numerator": 2,
              "status": "ok",
              "value": 0.6666666666666666
            },
            "precision": {
              "denominator": 1,
              "numerator": 1,
              "status": "ok",
              "value": 1.0
            },
            "recall": {
              "denominator": 2,
              "numerator": 1,
              "status": "ok",
              "value": 0.5
            },
            "support": 15
          },
          "favorable": {
            "accuracy": {
              "denominator": 15,
              "numerator": 15,
              "status": "ok",
              "value": 1.0
            },
            "confusion": {
              "fn": 0,
              "fp": 0,
              "tn": 13,
              "tp": 2
            },
            "f1": {
              "denominator": 4,
              "numerator": 4,
              "status": "ok",
              "value": 1.0
            },
            "precision": {
              "denominator": 2,
              "numerator": 2,
              "status": "ok",
              "value": 1.0
            },
            "recall": {
              "denominator": 2,
              "numerator": 2,
              "status": "ok",
              "value": 1.0
            },
            "support": 15
          },
          "unfavorable": {
            "accuracy": {
              "denominator": 15,
              "numerator": 12,
              "status": "ok",
              "value": 0.8
            },
            "confusion": {
              "fn": 3,
              "fp": 0,
              "tn": 12,
              "tp": 0
            },
            "f1": {
              "denominator": 3,
              "numerator": 0,
              "status": "ok",
              "value": 0.0
            },
            "precision": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "recall": {
              "denominator": 3,
              "numerator": 0,
              "status": "ok",
              "value": 0.0
            },
            "support": 15
          }
        },
        "confusion": {
          "fn": 6,
          "fp": 0,
          "tn": 50,
          "tp": 4
        },
        "f1": {
          "denominator": 14,
          "numerator": 8,
          "status": "ok",
          "value": 0.5714285714285714
        },
        "precision": {
          "denominator": 4,
          "numerator": 4,
          "status": "ok",
          "value": 1.0
        },
        "recall": {
          "denominator": 10,
          "numerator": 4,
          "status": "ok",
          "value": 0.4
        },
        "support": 60
      },
      "wants_whole_document": {
        "accuracy": {
          "denominator": 60,
          "numerator": 60,
          "status": "ok",
          "value": 1.0
        },
        "by_stratum": {
          "adversarial_boundary": {
            "accuracy": {
              "denominator": 15,
              "numerator": 15,
              "status": "ok",
              "value": 1.0
            },
            "confusion": {
              "fn": 0,
              "fp": 0,
              "tn": 15,
              "tp": 0
            },
            "f1": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "precision": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "recall": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "support": 15
          },
          "ambiguous": {
            "accuracy": {
              "denominator": 15,
              "numerator": 15,
              "status": "ok",
              "value": 1.0
            },
            "confusion": {
              "fn": 0,
              "fp": 0,
              "tn": 15,
              "tp": 0
            },
            "f1": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "precision": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "recall": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "support": 15
          },
          "favorable": {
            "accuracy": {
              "denominator": 15,
              "numerator": 15,
              "status": "ok",
              "value": 1.0
            },
            "confusion": {
              "fn": 0,
              "fp": 0,
              "tn": 15,
              "tp": 0
            },
            "f1": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "precision": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "recall": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "support": 15
          },
          "unfavorable": {
            "accuracy": {
              "denominator": 15,
              "numerator": 15,
              "status": "ok",
              "value": 1.0
            },
            "confusion": {
              "fn": 0,
              "fp": 0,
              "tn": 15,
              "tp": 0
            },
            "f1": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "precision": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "recall": {
              "denominator": 0,
              "numerator": 0,
              "status": "not_applicable",
              "value": null
            },
            "support": 15
          }
        },
        "confusion": {
          "fn": 0,
          "fp": 0,
          "tn": 60,
          "tp": 0
        },
        "f1": {
          "denominator": 0,
          "numerator": 0,
          "status": "not_applicable",
          "value": null
        },
        "precision": {
          "denominator": 0,
          "numerator": 0,
          "status": "not_applicable",
          "value": null
        },
        "recall": {
          "denominator": 0,
          "numerator": 0,
          "status": "not_applicable",
          "value": null
        },
        "support": 60
      }
    },
    "reason": {
      "by_label": {
        "collection_backed_question": {
          "conformance": {
            "denominator": 6,
            "numerator": 3,
            "status": "ok",
            "value": 0.5
          },
          "support": 6
        },
        "explicit_search": {
          "conformance": {
            "denominator": 13,
            "numerator": 9,
            "status": "ok",
            "value": 0.6923076923076923
          },
          "support": 13
        },
        "figure_request": {
          "conformance": {
            "denominator": 10,
            "numerator": 4,
            "status": "ok",
            "value": 0.4
          },
          "support": 10
        },
        "local_tool_request": {
          "conformance": {
            "denominator": 15,
            "numerator": 8,
            "status": "ok",
            "value": 0.5333333333333333
          },
          "support": 15
        },
        "no_retrieval_needed": {
          "conformance": {
            "denominator": 13,
            "numerator": 6,
            "status": "ok",
            "value": 0.46153846153846156
          },
          "support": 13
        },
        "retry_request": {
          "conformance": {
            "denominator": 3,
            "numerator": 3,
            "status": "ok",
            "value": 1.0
          },
          "support": 3
        }
      },
      "by_stratum": {
        "adversarial_boundary": {
          "by_label": {
            "collection_backed_question": {
              "conformance": {
                "denominator": 1,
                "numerator": 0,
                "status": "ok",
                "value": 0.0
              },
              "support": 1
            },
            "explicit_search": {
              "conformance": {
                "denominator": 3,
                "numerator": 2,
                "status": "ok",
                "value": 0.6666666666666666
              },
              "support": 3
            },
            "figure_request": {
              "conformance": {
                "denominator": 3,
                "numerator": 1,
                "status": "ok",
                "value": 0.3333333333333333
              },
              "support": 3
            },
            "local_tool_request": {
              "conformance": {
                "denominator": 4,
                "numerator": 3,
                "status": "ok",
                "value": 0.75
              },
              "support": 4
            },
            "no_retrieval_needed": {
              "conformance": {
                "denominator": 4,
                "numerator": 1,
                "status": "ok",
                "value": 0.25
              },
              "support": 4
            },
            "retry_request": {
              "conformance": {
                "denominator": 0,
                "numerator": 0,
                "status": "not_applicable",
                "value": null
              },
              "support": 0
            }
          },
          "conformance": {
            "denominator": 15,
            "numerator": 7,
            "status": "ok",
            "value": 0.4666666666666667
          },
          "confusion_matrix": {
            "collection_backed_question": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 1,
              "retry_request": 0
            },
            "explicit_search": {
              "collection_backed_question": 0,
              "explicit_search": 2,
              "figure_request": 0,
              "local_tool_request": 1,
              "no_retrieval_needed": 0,
              "retry_request": 0
            },
            "figure_request": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 1,
              "local_tool_request": 2,
              "no_retrieval_needed": 0,
              "retry_request": 0
            },
            "local_tool_request": {
              "collection_backed_question": 0,
              "explicit_search": 1,
              "figure_request": 0,
              "local_tool_request": 3,
              "no_retrieval_needed": 0,
              "retry_request": 0
            },
            "no_retrieval_needed": {
              "collection_backed_question": 0,
              "explicit_search": 1,
              "figure_request": 0,
              "local_tool_request": 2,
              "no_retrieval_needed": 1,
              "retry_request": 0
            },
            "retry_request": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 0,
              "retry_request": 0
            }
          },
          "labels": [
            "retry_request",
            "local_tool_request",
            "figure_request",
            "explicit_search",
            "collection_backed_question",
            "no_retrieval_needed"
          ],
          "support": 15
        },
        "ambiguous": {
          "by_label": {
            "collection_backed_question": {
              "conformance": {
                "denominator": 2,
                "numerator": 2,
                "status": "ok",
                "value": 1.0
              },
              "support": 2
            },
            "explicit_search": {
              "conformance": {
                "denominator": 2,
                "numerator": 2,
                "status": "ok",
                "value": 1.0
              },
              "support": 2
            },
            "figure_request": {
              "conformance": {
                "denominator": 2,
                "numerator": 1,
                "status": "ok",
                "value": 0.5
              },
              "support": 2
            },
            "local_tool_request": {
              "conformance": {
                "denominator": 4,
                "numerator": 2,
                "status": "ok",
                "value": 0.5
              },
              "support": 4
            },
            "no_retrieval_needed": {
              "conformance": {
                "denominator": 4,
                "numerator": 1,
                "status": "ok",
                "value": 0.25
              },
              "support": 4
            },
            "retry_request": {
              "conformance": {
                "denominator": 1,
                "numerator": 1,
                "status": "ok",
                "value": 1.0
              },
              "support": 1
            }
          },
          "conformance": {
            "denominator": 15,
            "numerator": 9,
            "status": "ok",
            "value": 0.6
          },
          "confusion_matrix": {
            "collection_backed_question": {
              "collection_backed_question": 2,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 0,
              "retry_request": 0
            },
            "explicit_search": {
              "collection_backed_question": 0,
              "explicit_search": 2,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 0,
              "retry_request": 0
            },
            "figure_request": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 1,
              "local_tool_request": 0,
              "no_retrieval_needed": 1,
              "retry_request": 0
            },
            "local_tool_request": {
              "collection_backed_question": 1,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 2,
              "no_retrieval_needed": 1,
              "retry_request": 0
            },
            "no_retrieval_needed": {
              "collection_backed_question": 2,
              "explicit_search": 1,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 1,
              "retry_request": 0
            },
            "retry_request": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 0,
              "retry_request": 1
            }
          },
          "labels": [
            "retry_request",
            "local_tool_request",
            "figure_request",
            "explicit_search",
            "collection_backed_question",
            "no_retrieval_needed"
          ],
          "support": 15
        },
        "favorable": {
          "by_label": {
            "collection_backed_question": {
              "conformance": {
                "denominator": 1,
                "numerator": 1,
                "status": "ok",
                "value": 1.0
              },
              "support": 1
            },
            "explicit_search": {
              "conformance": {
                "denominator": 5,
                "numerator": 5,
                "status": "ok",
                "value": 1.0
              },
              "support": 5
            },
            "figure_request": {
              "conformance": {
                "denominator": 2,
                "numerator": 2,
                "status": "ok",
                "value": 1.0
              },
              "support": 2
            },
            "local_tool_request": {
              "conformance": {
                "denominator": 3,
                "numerator": 3,
                "status": "ok",
                "value": 1.0
              },
              "support": 3
            },
            "no_retrieval_needed": {
              "conformance": {
                "denominator": 3,
                "numerator": 3,
                "status": "ok",
                "value": 1.0
              },
              "support": 3
            },
            "retry_request": {
              "conformance": {
                "denominator": 1,
                "numerator": 1,
                "status": "ok",
                "value": 1.0
              },
              "support": 1
            }
          },
          "conformance": {
            "denominator": 15,
            "numerator": 15,
            "status": "ok",
            "value": 1.0
          },
          "confusion_matrix": {
            "collection_backed_question": {
              "collection_backed_question": 1,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 0,
              "retry_request": 0
            },
            "explicit_search": {
              "collection_backed_question": 0,
              "explicit_search": 5,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 0,
              "retry_request": 0
            },
            "figure_request": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 2,
              "local_tool_request": 0,
              "no_retrieval_needed": 0,
              "retry_request": 0
            },
            "local_tool_request": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 3,
              "no_retrieval_needed": 0,
              "retry_request": 0
            },
            "no_retrieval_needed": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 3,
              "retry_request": 0
            },
            "retry_request": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 0,
              "retry_request": 1
            }
          },
          "labels": [
            "retry_request",
            "local_tool_request",
            "figure_request",
            "explicit_search",
            "collection_backed_question",
            "no_retrieval_needed"
          ],
          "support": 15
        },
        "unfavorable": {
          "by_label": {
            "collection_backed_question": {
              "conformance": {
                "denominator": 2,
                "numerator": 0,
                "status": "ok",
                "value": 0.0
              },
              "support": 2
            },
            "explicit_search": {
              "conformance": {
                "denominator": 3,
                "numerator": 0,
                "status": "ok",
                "value": 0.0
              },
              "support": 3
            },
            "figure_request": {
              "conformance": {
                "denominator": 3,
                "numerator": 0,
                "status": "ok",
                "value": 0.0
              },
              "support": 3
            },
            "local_tool_request": {
              "conformance": {
                "denominator": 4,
                "numerator": 0,
                "status": "ok",
                "value": 0.0
              },
              "support": 4
            },
            "no_retrieval_needed": {
              "conformance": {
                "denominator": 2,
                "numerator": 1,
                "status": "ok",
                "value": 0.5
              },
              "support": 2
            },
            "retry_request": {
              "conformance": {
                "denominator": 1,
                "numerator": 1,
                "status": "ok",
                "value": 1.0
              },
              "support": 1
            }
          },
          "conformance": {
            "denominator": 15,
            "numerator": 2,
            "status": "ok",
            "value": 0.13333333333333333
          },
          "confusion_matrix": {
            "collection_backed_question": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 2,
              "retry_request": 0
            },
            "explicit_search": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 3,
              "retry_request": 0
            },
            "figure_request": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 3,
              "retry_request": 0
            },
            "local_tool_request": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 4,
              "retry_request": 0
            },
            "no_retrieval_needed": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 1,
              "no_retrieval_needed": 1,
              "retry_request": 0
            },
            "retry_request": {
              "collection_backed_question": 0,
              "explicit_search": 0,
              "figure_request": 0,
              "local_tool_request": 0,
              "no_retrieval_needed": 0,
              "retry_request": 1
            }
          },
          "labels": [
            "retry_request",
            "local_tool_request",
            "figure_request",
            "explicit_search",
            "collection_backed_question",
            "no_retrieval_needed"
          ],
          "support": 15
        }
      },
      "conformance": {
        "denominator": 60,
        "numerator": 33,
        "status": "ok",
        "value": 0.55
      },
      "confusion_matrix": {
        "collection_backed_question": {
          "collection_backed_question": 3,
          "explicit_search": 0,
          "figure_request": 0,
          "local_tool_request": 0,
          "no_retrieval_needed": 3,
          "retry_request": 0
        },
        "explicit_search": {
          "collection_backed_question": 0,
          "explicit_search": 9,
          "figure_request": 0,
          "local_tool_request": 1,
          "no_retrieval_needed": 3,
          "retry_request": 0
        },
        "figure_request": {
          "collection_backed_question": 0,
          "explicit_search": 0,
          "figure_request": 4,
          "local_tool_request": 2,
          "no_retrieval_needed": 4,
          "retry_request": 0
        },
        "local_tool_request": {
          "collection_backed_question": 1,
          "explicit_search": 1,
          "figure_request": 0,
          "local_tool_request": 8,
          "no_retrieval_needed": 5,
          "retry_request": 0
        },
        "no_retrieval_needed": {
          "collection_backed_question": 2,
          "explicit_search": 2,
          "figure_request": 0,
          "local_tool_request": 3,
          "no_retrieval_needed": 6,
          "retry_request": 0
        },
        "retry_request": {
          "collection_backed_question": 0,
          "explicit_search": 0,
          "figure_request": 0,
          "local_tool_request": 0,
          "no_retrieval_needed": 0,
          "retry_request": 3
        }
      },
      "labels": [
        "retry_request",
        "local_tool_request",
        "figure_request",
        "explicit_search",
        "collection_backed_question",
        "no_retrieval_needed"
      ],
      "support": 60
    },
    "support": 60
  }
}
```

## Excluded claims

- No generated-answer correctness, relevance, faithfulness, or citation-entailment claim.
- No end-to-end latency, concurrency, GPU, or production-throughput claim.
- No authorization or database-isolation claim from syntax and prefix checks.
- No population estimate or sampling-based confidence interval.
