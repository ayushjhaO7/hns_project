import pandas as pd

df = pd.read_csv("data\crime_dataset.csv")

north = [
    "DELHI", "HARYANA", "PUNJAB", "HIMACHAL PRADESH",
    "JAMMU & KASHMIR", "LADAKH", "UTTAR PRADESH",
    "UTTARAKHAND", "RAJASTHAN", "CHANDIGARH"
]

south = [
    "ANDHRA PRADESH", "TELANGANA", "KARNATAKA",
    "KERALA", "TAMIL NADU", "PUDUCHERRY"
]

west = [
    "MAHARASHTRA", "GUJARAT", "GOA",
    "MADHYA PRADESH", "DADRA & NAGAR HAVELI",
    "DAMAN & DIU"
]

east = [
    "BIHAR", "JHARKHAND", "ODISHA", "WEST BENGAL",
    "ASSAM", "ARUNACHAL PRADESH", "MANIPUR",
    "MEGHALAYA", "MIZORAM", "NAGALAND", "TRIPURA",
    "SIKKIM"
]

df["STATE/UT"] = df["STATE/UT"].str.upper().str.strip()

def get_region(state):
    if state in north:
        return "North"
    elif state in south:
        return "South"
    elif state in west:
        return "West"
    elif state in east:
        return "East"
    else:
        return "Other"

df["Region"] = df["STATE/UT"].apply(get_region)

north_df = df[df["Region"] == "North"]
south_df = df[df["Region"] == "South"]
west_df  = df[df["Region"] == "West"]
east_df  = df[df["Region"] == "East"]

north_df.to_csv("north.csv", index=False)
south_df.to_csv("south.csv", index=False)
west_df.to_csv("west.csv", index=False)
east_df.to_csv("east.csv", index=False)