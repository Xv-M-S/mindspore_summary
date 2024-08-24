from setuptools import setup, find_packages

setup(
    name="netsummary",
    version="1.0.0",
    description="Torch has a neat API to view the visualization of the model which is very helpful while debugging your network. Here is a code to try and mimic the same in MindSpore. The aim is to provide information complementary to, what is not provided by print(your_model) in MindSpore.",
    url="https://github.com/Xv-M-S/mindspore_summary",
    author="shixumao",
    author_email="1205507925@qq.com",
    packages=["netsummary"],
)
