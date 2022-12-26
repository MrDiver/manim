window.BENCHMARK_DATA = {
  "lastUpdate": 1672067793743,
  "repoUrl": "https://github.com/MrDiver/manim",
  "entries": {
    "Manim Benchmark": [
      {
        "commit": {
          "author": {
            "name": "MrDiver",
            "username": "MrDiver"
          },
          "committer": {
            "name": "MrDiver",
            "username": "MrDiver"
          },
          "id": "24aa46f05acceb70a57d456e662fc93e381fce69",
          "message": "testing benchmarks",
          "timestamp": "2022-12-16T23:11:27Z",
          "url": "https://github.com/MrDiver/manim/pull/1/commits/24aa46f05acceb70a57d456e662fc93e381fce69"
        },
        "date": 1672067789645,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmark_mobject.py::test_benchmark_vmobject_add",
            "value": 0.06841226935077169,
            "unit": "iter/sec",
            "range": "stddev: 0.03344044841679792",
            "extra": "mean: 14.617261048199975 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_benchmark_mobject.py::test_benchmark_shift",
            "value": 186.4381292541265,
            "unit": "iter/sec",
            "range": "stddev: 0.002385245529180221",
            "extra": "mean: 5.363709687501419 msec\nrounds: 176"
          },
          {
            "name": "tests/benchmarks/test_benchmark_mobject.py::test_benchmark_become_single",
            "value": 29.092711718421345,
            "unit": "iter/sec",
            "range": "stddev: 0.0007934099583170191",
            "extra": "mean: 34.37287007408133 msec\nrounds: 27"
          },
          {
            "name": "tests/benchmarks/test_benchmark_mobject.py::test_benchmark_become_group",
            "value": 28.642781504628164,
            "unit": "iter/sec",
            "range": "stddev: 0.0007939257277026467",
            "extra": "mean: 34.91281040001013 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_benchmark_mobject.py::test_benchmark_render_scene",
            "value": 1.9632672979123567,
            "unit": "iter/sec",
            "range": "stddev: 0.008298261045355205",
            "extra": "mean: 509.35499260001507 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}