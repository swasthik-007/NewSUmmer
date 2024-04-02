/**
 * 
 * 1.write a function to download data from a url
 * 2.write a function to save that data un a file and return the filename
 * 3.write a function to upload the file written in previous step to a new url
 */
 function download(url){
    return new Promise(function exec(resolve,reject){
        console.log("starting to download from ",url);
    setTimeout(function down(){
        console.log("downloading completed");
        const content="ABCDEF"; // assume dummy download content
        resolve(content);
    },6000);
    })
 }

 function writefile(data){
    return new Promise(function exec(resolve,reject){
        console.log("started writing a file with", data);
        setTimeout(function write(){
            console.log("comlpeted writing the data in a file");
            const filename="file.txt";
            resolve(filename);
        },5000);
    })
 }

 function upload(file,url){
    return new Promise(function exec(resolve,reject){
        console.log("started uploading ",file ,"on",url);
    setTimeout(function up(){
        console.log("upload competed");
        const response="success";
        resolve(response);
    },2000);
    } )
 }

 download("www.xyz.com")
 .then(function processdownload(value){
    console.log("downloading the data with the following value",value);
return writefile(value);
 })
 .then(function processwrite(value){
    console.log("data is written with the file name",value);
    return upload(value,"www.upload.com");
 })
 .then(function processupload(value){
    console.log("we have uploaded with a value",value);

 })

