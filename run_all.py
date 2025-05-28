#!/bin/python
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.common_functions import launch_collection_of_configs, parse_args


# ========== Launching scripts ==========
def run_stuff(field):
    msg = f"LAUNCHING {field} with {num_chs} CHANNELS!".center(40)
    msg = f"°`°º¤ø,,ø¤°º¤ø,,ø¤º°`° {msg} °`°º¤ø,,ø¤°º¤ø,,ø¤º°`°".center(100)
    print(f"\n{msg}\n")

    name = f"{num_chs}ch/QML_{field}_{num_chs}ch"
    notens_name = f"{name}_notens"
    types = [
        "",
        "_mismatch",
        "_mismatch_2",
        "_wrongfid",
        "_wrongfid_2",
    ]

    names = [f"{name}{t}_config.yaml" for t in types]
    if want_notens and field == "BB":
        names += [f"{notens_name}{t}_config.yaml" for t in types]

    configs_collection = [config_dir + n for n in names]
    launch_collection_of_configs(configs_collection)

    msg = "ALL DONE!".center(40)
    msg = f"°`°º¤ø,,ø¤°º¤ø,,ø¤º°`° {msg} °`°º¤ø,,ø¤°º¤ø,,ø¤º°`°".center(100)
    print(f"\n{msg}\n")


if __name__ == "__main__":
    parsed_args = parse_args()
    config_path = parsed_args.config_path
    field = parsed_args.field
    num_chs = parsed_args.num_chs
    want_notens = parsed_args.want_notens
    config_dir = os.path.dirname(os.path.abspath(config_path)) + "/configs/"

    start = time.time()
    if field == "ALL":
        run_stuff("BB")
        run_stuff("EE")
    elif field in ["BB", "EE"]:
        run_stuff(field)
    else:
        raise ValueError(
            f"Invalid field '{field}'. Please choose 'BB', 'EE', or 'ALL'."
        )
    end = time.time()
    print(f"Total time taken: {(end - start) / 60:.2f} minutes!")
