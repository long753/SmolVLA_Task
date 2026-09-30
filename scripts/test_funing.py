import argparse

# 定义实验室名单
names = ["Alice", "Bob", "Charlie", "David", "Eve"]
ages = [25, 30, 35, 40, 45]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="default",
        help="Name for the experiment.",
    )
    parser.add_argument(
        "--age",
        type=int,
        default=0,
        help="Age for the experiment.",
    )

    for name in names:
        if name == parser.parse_args().name:
            print(f"Name: {name}, Age: {ages[names.index(name)]}")
            break
    else:
        print("Name not found in the list.")
    
    
