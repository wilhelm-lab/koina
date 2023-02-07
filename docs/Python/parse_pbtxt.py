#!/usr/bin/env python3

"""
by

#EuBIC2023 Monte Verità, Switzerland
https://eubic-ms.org/events/2023-developers-meeting/


https://github.com/eubic/EuBIC2023/issues/12

this script takes as input a .pbtxt file
and generates GraphViz .dot output


usage:
  parse_pbtxt.py config.pbtxt
"""

import sys
import os

"""
parse pbtxt file

TODO(cp):
- rename es to ensemble_scheduling
- rename ri to reverseIndex
- refactor states 
"""


def parseConfig(file="config.pbtxt", debug=False):
    es = {}
    ri = {}

    inputNode = {}
    outputNode = {}

    ln = 0
    state = 0
    modelName = ""
    inputName = ""
    outputName = ""

    with open(file, "r") as f:
        for line in f:
            line = line.strip().replace('"', "").replace("'", "").replace(",", "")
            ln += 1

            if debug:
                print("{}\t{}".format(ln, line))

            if "model_name:" in line:
                key, modelName = line.replace('"', "").split(": ")
                es[modelName] = {"input": [], "output": []}

            if "input_map" in line or "ensemble_scheduling" in line:
                state = 0

            if "output_map" in line:
                state = 1

            if "input [" in line:
                state = 2

            if "output [" in line:
                state = 3

            if state == 2 and "name:" in line:
                key, value = line.split(": ")
                inputName = value
                inputNode[inputName] = {}

            if state == 2 and ("data_type:" in line or "dims:" in line):
                key, value = line.split(": ")
                inputNode[inputName][key] = value

            if state == 3 and "name:" in line:
                key, value = line.split(": ")
                outputName = value
                outputNode[outputName] = {}

            if state == 3 and ("data_type:" in line or "dims:" in line):
                key, value = line.split(": ")
                outputNode[outputName][key] = value

            if state == 0 and "key:" in line:
                key, value = line.split(": ")
                es[modelName]["input"].append("in:key:" + value)

            if state == 0 and "value:" in line:
                key, value = line.split(": ")
                es[modelName]["input"].append("in:value:" + value)

            if state == 1 and "key:" in line:
                key, value = line.split(": ")
                es[modelName]["output"].append("out:key:" + value)

            if state == 1 and "value:" in line:
                key, value = line.split(": ")
                es[modelName]["output"].append("out:value:" + value)

                ri["in:value:" + value] = modelName

    return (inputNode, es, ri, outputNode)


"""
compose GraphViz dot file

TODO:
- make the subgraph with grey background
- use LR assignment for labels in records
"""


def composeGraphViz(input, label="ES"):
    inputNodes, es, ri, outputNodes = input
    print("digraph G {")

    dot_pre = """
    fontname="Helvetica,Arial,sans-serif"
    node [fontname="Helvetica,Arial,sans-serif"]
    edge [fontname="Helvetica,Arial,sans-serif"]
    graph [ rankdir = "LR" ];
    node [ fontsize = "16" shape = "ellipse" ];
    edge [ ];
    """
    print(dot_pre)

    print("## {}".format(es))
    print("## {}".format(inputNodes))
    print("## {}".format(outputNodes))

    # draw models

    print("\tsubgraph ensemble0 {\nstyle=filled;\ncolor=lightgrey;\n")
    edgeTmp = ""
    for k in es:
        print("""\t\t"{}" [\n\t\t\tlabel="<f0> {}""".format(k, k), end="")
        for i in es[k]["input"]:
            if not i.startswith("in:key"):
                print(""" | <{}> {}""".format(i.replace(":", "_"), i), end="")

            if "in:value:" in i and i in ri:
                edgeTmp = edgeTmp + """\t\t"{}":"{}" -> "{}":"{}";\n""".format(
                    ri[i],
                    i.replace("in:value", "out:value").replace(":", "_"),
                    k,
                    i.replace(":", "_"),
                )
        for i in es[k]["output"]:
            ## TODO XX
            if not i.startswith("out:key"):
                print("""| <{}> {}""".format(i.replace(":", "_"), i), end="")
        print(""""\n\t\tshape="record"\n\t\t];""")

    print("\n\n{}".format(edgeTmp))

    # subgraph
    print(""" label = <<b>Prosit intensity 2019</b>>; """)
    print("}")

    print("\tsubgraph input_0 {")
    print("""\t\tstyle=filled;""")
    print("""\t\tnode [style=filled, color=lightgrey];""")
    for k in inputNodes:
        print("""\t"{}" [ label="<i0> {}""".format(k.replace(":", "_"), k), end="")
        print(
            """| <input> {} {}""".format(
                inputNodes[k]["data_type"], inputNodes[k]["dims"]
            ),
            end="",
        )
        print("""" shape="record"];""")
    for k in es:
        for i in es[k]["input"]:
            if "in:value:" in i and not i in ri:
                print(
                    """\t\t"{}":"{}" -> "{}":"{}"\n""".format(
                        i.replace(":", "_").replace("in_value_", ""),
                        "input",
                        k,
                        i.replace(":", "_"),
                    )
                )
    print("\t}")

    print("\tsubgraph output_0 {")
    print("""\t\tstyle=filled;""")
    print("""\t\tnode [style=filled,color=lightgrey];""")
    print("""\t\tlabel = "output";""")

    for k in outputNodes:
        print(""""{}" [\nlabel="<o0> {}""".format(k.replace(":", "_"), k), end="")
        print(
            """| <output> {} {}""".format(
                outputNodes[k]["data_type"], outputNodes[k]["dims"]
            )
        )
        print(""""\nshape="record"\n];""")

    for k in es:
        for i in es[k]["output"]:
            out = i.replace("out:value:", "")
            if out in outputNodes:
                print(
                    """\t\t"{}":"{}" -> "{}":output""".format(
                        k, i.replace(":", "_"), out.replace(":", "_")
                    )
                )

    print("\t}")
    print("}")


### MAIN

# TODO: could be used for unit testing
# f = "/Users/cp/__checkouts/dlomix-serving//models/Prosit/Prosit_2019_intensity_triton/config.pbtxt"
# f = "/Users/cp/__checkouts/dlomix-serving//models/Deeplc/Deeplc_Triton_ensemble/config.pbtxt"
# f = "ludwig_config.pbtxt"
f = sys.argv[1]
if os.path.isfile(f):
    ensemble_scheduling = parseConfig(f, debug=False)
    composeGraphViz(ensemble_scheduling, label=f)
else:
    raise ("File '{}' does not exist.".format(f))
