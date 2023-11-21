import argparse
from data import *

parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset', default="agedb", type=str,
                    help='datasets requiring interval data generation')
parser.add_argument('--max_interval', default=40, type=float,
                    help='q: maximum allowable value of the interval size')
parser.add_argument('--yl', default=-50, type=float,
                    help='Minimum value of interval')
parser.add_argument('--yr', default=200, type=float,
                    help='Maximum value of interval')


def main():
    args = parser.parse_args()
    if args.dataset == "abalone":
        abalone_data_processing(args.max_interval, args.yl, args.yr)
    elif args.dataset == "airfoil":
        airfoil_data_processing(args.max_interval, args.yl, args.yr)
    elif args.dataset == "auto-mpg":
        auto_mpg_data_processing(args.max_interval, args.yl, args.yr)
    elif args.dataset == "housing":
        housing_data_processing(args.max_interval, args.yl, args.yr)
    elif args.dataset == "concrete":
        concrete_data_processing(args.max_interval, args.yl, args.yr)
    elif args.dataset == "power-plant":
        power_plant_data_processing(args.max_interval, args.yl, args.yr)
    elif args.dataset == "agedb":
        AgeDB_data_processing(args.max_interval, args.yl, args.yr)
    elif args.dataset == "imdb-wiki":
        imdb_wiki_data_processing(args.max_interval, args.yl, args.yr)


if __name__ == '__main__':
    main()
