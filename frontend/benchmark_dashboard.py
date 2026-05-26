import streamlit as st

from app.evaluation.benchmark import Benchmark


class BenchmarkDashboard:

    def show(self):

        data = Benchmark().benchmark()

        st.subheader("Benchmark Dashboard")

        st.write(data)