/**
 * Tasks: (Don't use promises, only use callbacks)
 * 1.write a function to download data from a url
 * 2.write a function to save that data un a file and return the filename
 * write a function to upload the file written in previous step to a new url
 */

function download(url, cb){
    /**
     * downloads content fromt the given url and passes the downloaded contebt 
     * to the given callback
     */
    console.log("starting to download from url",url);
    setTimeout(function down(){
        console.log("downloading completed");
        const content="ABCDEF"; // assume dummy download content
        cb(content);
    },4000);
}

function writefile(data,cb){
    console.log("started writing a file with", data);
    setTimeout(function write(){
        console.log("comlpeted writing the data in a file");
        const filename="file.txt";
        cb(filename);
    },5000);

}

function upload(url,file, cb){
    console.log("started uploading ",file ,"on",url);
    setTimeout(function up(){
        console.log("upload competed");
        const response="success";
        cb(response);
    },2000)
}

// download("www.asc.com",function process(content){
//     console.log("the data is ",content);
// })

// writefile("abcdef",function process(name){
//     console.log("file written with a name ",name);
// })/

download("www.xyz.com", function proccessDownload(content){
    console.log("we are now going to process the downloaded data ");
    writefile(content,function processwrite(filename){
        console.log("we have downloaded and written the file, now will upload");
        upload("www.upload.com",filename,function processUpload(response){
            console.log("we have uploaded with",response);
        });
    });
});