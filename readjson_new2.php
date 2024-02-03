<?php

ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);
ini_set('memory_limit', '400M');

$drugs = $_REQUEST['drugs'];
$diseases = $_REQUEST['diseases'];

$command = "/usr/bin/Rscript filter_data_combi.R '$drugs' '$diseases'";

// Execute the R script and store the output
$output = shell_exec($command);

// Output the result
echo $output;


/*
// Convert to array 
$array = json_decode($strJsonFileContents, true);

// Print array
print_r($array);
*/

?>