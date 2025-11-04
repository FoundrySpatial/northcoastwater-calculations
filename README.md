# North Coast Water Availability Tool Calculations
Calculation Files For the California North Coast Water Availability Tool (northcoastwater.codefornature.org). Contained within here is all of the logic used within the tool to generate Water Supply Reports and Cumulative Diversion Analyses.

## Directory structure:

The following is the directory structure of this repository. The "api" folder represents code that lives in the tool API that performs calculations, the "documentation" folder contains static PDF documentation that is in the tool and output packages and the "sqitch" folder contains database (.sql) files. Each of these are expanded upon below.

```
├── api
│   ├── queries
│   ├── routers
│   ├── utils
│   └── database.py
├── documentation
└── stored_procedures

```

## API

The API Folder Contains code that lives in the Tool API and runs calculations. For the most part, all of these calculations are performed upon request, and data that is user-entered and from the database is used to create the output data.

### Queries Folder

The queries folder under /api/queries contains some queries used to interact with the database to generate output data. A few examples of what is done here:

- Simple insert to create user project data
- GIS operations using PostGIS to generate watersheds and perform raster lookups on watershed polygons
- Queries to output user summary data and perform some small data migrations on the data

### Utils Folder

The utils folder under /api/utils contains some utility functions that make up the bulk of the calculation logic in the api. In general, the python data processing library [pandas](https://pandas.pydata.org/ "pandas") is used throughout these utility functions to perform data processing operations such as transformations, group by operations, iteration and vectorized manipulations.

The files within the utils folder are :

1. api_utils.py - general utility functions for the API
2. cda_output_package.py - functionality to generate and email a user their CDA (cumulative diversion analysis) output package.
3. cda_utils - general functionality to implement the algorithms and data transformations for the Cumulative Diversion Analysis
4. helpers.py - Some utility helper functions for simple things like unit conversions, etc.
5. wsr_csv_utils.py - general functionality to implement algorithms and data transformations for the Water Supply Report. Furthermore, this is responsible for handling the upload and validation of user-uploaded csv's.

### Routers Folder

The routers folder under /api/routers defines the routes (api endpoints) of the tool API. Each of these routes follow a general pattern:

1. Get data from database (using a query in the queries folder or a stored procedure)
2. Use a utility function to format data and perform calculations (if necessary)
3. Return data formatted (to the running site)

The routers are as follows:

1. gages.py - Get gage date from USGS gauges
2. points_of_diversion.py - get data about existing points of diversion in the north coast area
3. projects.py - perform project-specific handling of data such as generating project output files and tables viewed in the tool
4. search.py - search functionality for maps
5. stream_reach.py - get stream reach data and verify it is in the correct area
6. streampaths.py - get streampath data and verify it is in the correct area
7. watersheds.py - get watersheds information from streams

### Database

The database.py and db_utils.py files define interactions with the postgreSQL database used by the tool. There are a variety of functionalities in database.py, but there are 2 main ways that the database is interacted with:

1. Through a query from queries.py - the query is imported from the above queries folder
2. Through a postgresSQL stored procedure - these are stored in the database

See the `stored_procedures` section below for more information about stored procedures.

## Documentation

The Documentation included in the tool is included here for reference and context about any calculations performed. This documentation can be a useful first step in viewing the functionality and can be a touchstone when viewing code as to the "bigger picture" of what the code is trying to accomplish.

## Stored Procedures

PostgreSQL stored procedures are used in a few places in this application to perform complex database operations and data formatting/transformations. Many of the stored procedures are trivial, but those with meaningful calculations have been included in the stored_procedures directory. These are referenced from the api/database.py file, and therefore their outputs are used in the data processing of the API routers.
