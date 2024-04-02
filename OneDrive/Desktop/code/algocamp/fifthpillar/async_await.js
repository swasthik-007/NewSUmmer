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

 function uploaddata(file,url){
    return new Promise(function exec(resolve,reject){
        console.log("started uploading ",file ,"on",url);
    setTimeout(function up(){
        console.log("upload competed");
        const response="success";
        resolve(response);
    },2000);
    } )
 }
 async function steps(){
    const downloadeddata=await download("www.xyz.com");
    console.log("data donwloaded is ",downloadeddata);
    const filewritten=await writefile(downloadeddata);
    console.log("file written is",filewritten);
    const uploadresponse =await uploaddata(filewritten,"www.drive.google.com");
    console.log("upload response is ",uploadresponse);
    return uploadresponse;

} 
steps();
console.log("outside")