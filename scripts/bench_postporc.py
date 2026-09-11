#!/usr/bin/env python3

import time
import subprocess
import argparse
import json


def run_cmd(cmd):

    start=time.perf_counter()

    subprocess.run(
        cmd,
        shell=True,
        check=True
    )

    return time.perf_counter()-start



def main():

    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--input"
    )

    parser.add_argument(
        "--output"
    )

    args=parser.parse_args()


    result={}


    # VAD placeholder
    # сюда подключается ваш VAD module

    start=time.perf_counter()

    # vad_result = vad(args.input)

    result["vad_sec"] = (
        time.perf_counter()-start
    )


    # concat

    result["concat_sec"] = run_cmd(
        f"""
        ffmpeg -y \
        -i {args.input} \
        -c copy \
        {args.output}
        """
    )


    # subtitles burn

    result["subtitle_sec"] = run_cmd(
        f"""
        ffmpeg -y \
        -i {args.output} \
        -vf subtitles=subs.srt \
        final.mp4
        """
    )


    result["S"] = sum(
        result.values()
    )


    print(
        json.dumps(
            result,
            indent=2
        )
    )


if __name__=="__main__":
    main()