from setuptools import setup

setup(
    name='otomi-segmenter',
    version='0.1',
    py_modules=['segmenter'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        otomi-segmenter=segmenter:cli
    ''',
)
