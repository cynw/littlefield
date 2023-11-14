import random
from statistics import NormalDist

import numpy as np
import pandas as pd
import simpy
import streamlit as st

PROCESS_TIMES_HR = [8.3, 0.8, 0.4, 0.1]
STATION_PRIORITY = [1, 1, 1, 1]
HOUR_PER_DAY = 24
PHASES = [60, 240, 300]
MAX_DAY = 360

PRICE_POLICY = [{
    "sale": 200.0,
    "quoted_lead_time": 7.0,
    "max_delay_time": 7.0
}, {
    "sale": 225.0,
    "quoted_lead_time": 1.0,
    "max_delay_time": 6.0
},
    {
        "sale": 250.0,
        "quoted_lead_time": 0.25,
        "max_delay_time": 0.75
    }
]

job_history = []
today_demand = 0
demand_historical = []

wip = 0
wip_history = []

st.title('LittleField simulator')


def job(env, name, stations):
    start_time = env.now
    station_end_time = []

    global wip
    wip_history.append([start_time, wip])

    # Do not start the job if wip is over the limit
    if wip > wip_limit:
        job_history.append([name, start_time] + [np.NaN, np.NaN, np.NaN, np.NaN] + [np.NaN, 0, 0, 0])
    else:
        wip += 1
        for idx, station in enumerate(stations):
            with station.request(priority=STATION_PRIORITY[idx]) as req:
                yield req
                yield env.timeout(PROCESS_TIMES_HR[idx] / HOUR_PER_DAY)
            station_end_time.append(float(env.now))
        flow_time = float(env.now - start_time)
        revenues = list(map(lambda policy: max(0, policy["sale"] -
                                               (max(0, flow_time - policy["quoted_lead_time"]) * policy["sale"] /
                                                policy["max_delay_time"])),
                            PRICE_POLICY))
        job_history.append([name, start_time] + station_end_time + [flow_time] + revenues)
        wip -= 1


start_day = st.sidebar.number_input('Start day', min_value=1, max_value=MAX_DAY, value=1)
end_day = st.sidebar.number_input('End day', min_value=1, max_value=MAX_DAY, value=1)

wip_limit = st.sidebar.number_input('WIP Limit (jobs)', min_value=0, max_value=6000, value=1000)

st.sidebar.title("New Machines")
s1 = st.sidebar.number_input('N Station 1', min_value=1, max_value=50, value=1)
s2 = st.sidebar.number_input('N Station 2', min_value=1, max_value=50, value=1)
s3 = st.sidebar.number_input('N Station 3', min_value=1, max_value=50, value=1)

st.sidebar.title("Previous Machines")
previous_s1 = st.sidebar.number_input('P Station 1', min_value=1, max_value=50, value=1)
previous_s2 = st.sidebar.number_input('P Station 2', min_value=1, max_value=50, value=1)
previous_s3 = st.sidebar.number_input('P Station 3', min_value=1, max_value=50, value=1)

st.sidebar.title("Others")

station3_priority = st.sidebar.radio("Priority at station2", ["Station 1", "Station 3"])
if station3_priority == "Station 1":
    STATION_PRIORITY[1] = 0
else:
    STATION_PRIORITY[3] = 0

cv = st.sidebar.number_input("Variation", min_value=0, max_value=20, value=2)
if cv == 0:
    cv = 0.000001

demand_intercept = st.sidebar.number_input("Demand Curve Intercept", value=2.2989)
p1_slope = st.sidebar.number_input("Phase 1 Slope", value=0.1377)
p2_slope = st.sidebar.number_input("Phase 2 Slope", value=0.1377 * 2)

env = simpy.Environment()
station1 = simpy.PriorityResource(env, capacity=s1)
station2 = simpy.PriorityResource(env, capacity=s2)
station3 = simpy.PriorityResource(env, capacity=s3)

station_maps = [station1, station2, station3, station2]


def setup(env):
    global start_day
    global end_day
    yield env.timeout(start_day)

    demand_historical = []

    def phase1(day):
        return demand_intercept + (day * p1_slope)

    def phase2(day):
        return (day * p2_slope) - phase1(PHASES[0]) + demand_intercept + phase1(1)

    def phase3(day):
        return phase2(PHASES[1])

    def phase4(day):
        return max(0.0, phase3(PHASES[2]) - (day - PHASES[2]) * phase3(PHASES[2]) / (MAX_DAY - PHASES[2]))

    def random_demand(d):
        return int(NormalDist(mu=d, sigma=cv).inv_cdf(random.random()))

    for d1 in range(max(1, start_day), min(end_day, PHASES[0]) + 1):
        demand_historical.append(phase1(d1))

    for d2 in range(max(PHASES[0] + 1, start_day), min(end_day, PHASES[1]) + 1):
        demand_historical.append(phase2(d2))

    for d3 in range(max(PHASES[1] + 1, start_day), min(end_day, PHASES[2]) + 1):
        demand_historical.append(phase3(d3))

    for d4 in range(max(PHASES[2] + 1, start_day), min(end_day, MAX_DAY) + 1):
        demand_historical.append(phase4(d4))

    # Add variability to demand
    demand_historical = list(map(random_demand, demand_historical))

    job_id = 1
    for demand in demand_historical:
        for idx in range(demand):
            env.process(job(env, 'Job %d' % job_id, station_maps))
            job_id += 1
        yield env.timeout(1)


env.process(setup(env))
env.run()


def machine_purchase_cost():
    global cost_of_machine
    cost_of_machine = 0
    if s1 > previous_s1:
        cost_of_machine += 25000 * (s1 - previous_s1)
    else:
        cost_of_machine -= 10000 * (previous_s1 - s1)
    if s2 > previous_s2:
        cost_of_machine += 75000 * (s2 - previous_s2)
    else:
        cost_of_machine -= 10000 * (previous_s2 - s2)
    if s3 > previous_s3:
        cost_of_machine += 75000 * (s3 - previous_s3)
    else:
        cost_of_machine -= 10000 * (previous_s3 - s3)
    return cost_of_machine


# A user must press a streamlit button to show the result
if st.button("Run simulation"):
    job_history_df = pd.DataFrame(job_history,
                                  columns=['Name', 'start_time', 'Station1_end_time', 'Station2_end_time',
                                           'Station3_end_time', 'Station4_end_time', 'flow_time', 'rev1',
                                           'rev2', 'rev3'])
    demand_historical_df = pd.DataFrame(demand_historical, columns=['demand'])
    cost_of_machine = machine_purchase_cost()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Job", len(job_history_df.index))
        st.subheader('Total Rev')
        st.metric("P1", int(job_history_df['rev1'].sum()))
        st.metric("P2", int(job_history_df['rev2'].sum()))
        st.metric("P3", int(job_history_df['rev3'].sum()))

    with col2:
        st.metric("Purchase machine", int(machine_purchase_cost()))
        st.subheader('Total Rev - Cost of Machine')
        st.metric("P1", int(job_history_df['rev1'].sum() - cost_of_machine))
        st.metric("P2", int(job_history_df['rev2'].sum() - cost_of_machine))
        st.metric("P3", int(job_history_df['rev3'].sum() - cost_of_machine))

    st.subheader('All jobs')
    st.dataframe(job_history_df)
    st.subheader('Stat')
    st.dataframe(job_history_df.describe())
    st.subheader('Demand (jobs)')
    st.line_chart(job_history_df.groupby("start_time").count(), y='Name')
    st.subheader('Avg flow time (days)')
    st.line_chart(job_history_df.groupby("start_time").mean(numeric_only=True), y='flow_time')

    wip_df = pd.DataFrame(wip_history, columns=['start_time', 'wip'])
    st.subheader('WIP (jobs)')
    st.line_chart(wip_df.groupby("start_time").mean(numeric_only=True), y='wip')
