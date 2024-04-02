function download(url){
    return new Promise(function exec(resolve,reject){
        console.log("starting to download from ",url);
    setTimeout(function down(){
        console.log("downloading completed");
        const content="ABCDEF"; // assume dummy download content
        reject(content);
    },6000);
    })
 }



async function steps(){
 try {
    console.log("starting steps");
const downloadeddata=await download("www.xyz.com");

return downloadeddata;
 } catch (error) {
    console.log("we have handled the error",error);
 }  
 finally{
    console.log("ending");
 }
}

steps();
 