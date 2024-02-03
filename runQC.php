<?php
echo shell_exec('/usr/bin/Rscript server1.R '.$_REQUEST['name'].' "'.$_REQUEST['smiles'].'" 2>&1'); 
?>