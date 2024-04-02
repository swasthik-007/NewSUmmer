document.addEventListener("DOMContentLoaded",()=>{
    async function populateproduct(){
        const queryparams=getqueryparams();
        if(queryparams['id']){
            const productid=queryparams['id'];
            const product=await fetchproductbyid(productid);
            

            const productname=document.getElementById('productname');
            const productprice=document.getElementById('productprice');
            const productdesc=document.getElementById('productdescriptiondata');
            const productimg=document.getElementById('productimg');
        
            productname.textContent=product.title;
            productdesc.textContent=product.description;
            productimg.src=product.image;
            productprice.textContent=product.price;
            removeloader();
        }
    }
    populateproduct();
})