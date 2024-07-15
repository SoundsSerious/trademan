#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['PyPortfolioOpt','randomname','numpy','scipy','cvxpy','yfinance','matplotlib','diskcache','scikit-learn']

test_requirements = [ ]

setup(
    author="Kevin Russell",
    author_email='kevin@ottermatics.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="trade and analyize stocks",
    entry_points={
        'console_scripts': [
            #'trademan=trademan.cli:main',
            'market_dl=trademan.data:main',
            'trademan=trademan.portfolio:cli'
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='trademan',
    name='trademan',
    packages=find_packages(include=['trademan', 'trademan.*']),
    package_data={'trademan':['*.csv'],
                  'trademan.media':['*.png']},
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/soundsserious/trademan',
    version='0.1.0',
    zip_safe=False,
)
