import json
import os
import argparse
from typing import Optional
from dotenv import load_dotenv
from tools.script import CurveFitting

# load env (.env optional)
load_dotenv()

def parse_args():
    p = argparse.ArgumentParser(description="Curve fitting agent runner")
    p.add_argument("--data_path", default="PEASnPbI4-Toluene.csv")
    p.add_argument("--comp_file", default="2D-3D (1).csv")
    p.add_argument("--start_wavelength", type=int, default=500)
    p.add_argument("--end_wavelength", type=int, default=850)
    p.add_argument("--wavelength_step_size", type=int, default=1)
    p.add_argument("--time_step", type=int, default=100)
    p.add_argument("--number_of_reads", type=int, default=100)
    p.add_argument("--reads", type=str, default="1-100", help="e.g. '1-100', 'odd', 'even', or '1,5,7'")
    p.add_argument("--wells_to_ignore", type=str, default="", help="comma-separated well ids, e.g. A1,B2")
    return p.parse_args()


def parse_reads_spec(spec: str, default_max: int) -> list:
    spec = spec.strip().lower()
    if spec == "odd":
        return [i for i in range(1, default_max+1) if i % 2 == 1]
    if spec == "even":
        return [i for i in range(1, default_max+1) if i % 2 == 0]
    if "-" in spec:
        a, b = spec.split("-", 1)
        a = int(a.strip()); b = int(b.strip())
        return list(range(min(a,b), max(a,b)+1))
    # comma-separated
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def main():
    args = parse_args()

    # Instantiate Curve Fitting Agent with user-configured parameters
    lum_reads = parse_reads_spec(args.reads, args.number_of_reads)
    curve_agent = CurveFitting(
        start_wavelength=args.start_wavelength,
        end_wavelength=args.end_wavelength,
        wavelength_step_size=args.wavelength_step_size,
        time_step=args.time_step,
        number_of_reads=args.number_of_reads,
        luminescence_read_numbers=lum_reads,
        wells_to_ignore=[w.strip() for w in args.wells_to_ignore.split(",") if w.strip()] if args.wells_to_ignore else ""
    )

    # Define path to curve data and comp file
    data_path = args.data_path
    comp_file = args.comp_file

    # Run analysis
    result = curve_agent.analyze_curve_fitting(data_path=data_path, comp_path=comp_file)

    # Print results
    print(f"--- Fitting and Analysis Summary ---\n")
    print(result)

    print("\n--- Fitted Parameters ---\n")
    print(json.dumps(result.get("fitting_parameters", {}), indent=2 ))

if __name__ == "__main__":
    main()


