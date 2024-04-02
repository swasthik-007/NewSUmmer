/**
 *  # Browser fetch the hrml page
 *  # begins parsing the html page
 *  # while parsing it encounters a script tag refering to external file
 *  # browser requests the external files and blocks the parser , hence parsing of html is halted
 *  # once the script is downloaded it is executed subsequently and parser restarts
 *  #any script(js) is capable to insert its own html doc using various function . that means the parser needs to wait yntiil the  script is not just downloaded but also executed  
 */ 


