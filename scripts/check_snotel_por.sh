#!/bin/bash
# This script generates and attempts to open a url link to the period of record SNOTEL depth data for input station IDs.
# Sample url to construct
# https://wcc.sc.egov.usda.gov/reportGenerator/view/customChartReport/daily/start_of_period/717:CO:SNTL%257C825:CO:SNTL%257C940:CO:SNTL%257C913:CO:SNTL%257Cid=%2522%2522%257Cname/POR_BEGIN,POR_END/stationId,name,SNWD::value?useLogScale=false
# Middle part that changes is this
# 717:CO:SNTL%257C825:CO:SNTL%257C940:CO:SNTL%257C913:CO:SNTL%257Cid=%2522%2522%257C

# split out the URL encoding for the pipe character

urlencoding='%257C'
# pipe is %7C
# percent sign is %25
# So this is the url encoding for a percent sign followed by a pipe

# Base URL components
begin='https://wcc.sc.egov.usda.gov/reportGenerator/view/customChartReport/daily/start_of_period/'
end='name/POR_BEGIN,POR_END/stationId,name,SNWD::value?useLogScale=false'

# Take user input argument as a list of station IDs
if [ $# -eq 0 ]; then
    echo "Usage: $0 station_id1 station_id2 ..."
    exit 1
fi

# Place default user input for state abbreviation
state_abbr='CO'

# Construct the "middle" string of sitenumbers and state abbreviation for the URL
middle=''
for station_id in "$@"; do
    # Check if the station ID is valid (e.g., numeric)
    if [[ ! $station_id =~ ^[0-9]+$ ]]; then
        echo "Invalid station ID: $station_id"
        exit 1
    fi
    # Append the station ID and state abbreviation to the middle string
    middle+="${station_id}:${state_abbr}:SNTL${urlencoding}"
done

# Generate the full encoding
full_url="${begin}${middle}id=%22%22%257C${end}"
echo "Generated URL: $full_url"

# Open the URL in the default web browser
if command -v xdg-open &> /dev/null; then
    xdg-open "$full_url"
elif command -v open &> /dev/null; then
    open "$full_url"
else
    echo "No suitable command found to open the URL. Please open it manually: $full_url"
fi
