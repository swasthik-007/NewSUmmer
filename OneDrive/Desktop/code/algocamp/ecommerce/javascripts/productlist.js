document.addEventListener("DOMContentLoaded",() =>{
     async function fetchproducts(){
        const response=await axios.get("https://fakestoreapi.com/products");
        console.log(response.data);
        return response.data;
    
    }
    async function populateproducts(){
        const products =await fetchproducts();
        const productlist=document.getElementById("productlist");
        products.forEach(product => {
            const productitem=document.createElement("a");
            productitem.target="_blank";
            productitem.classList.add("product-item","text-decoration-none","d-inline-block");
            productitem.href="productdetails.html";

            const productimage=document.createElement("div");
            const productname=document.createElement("div");
            const productprice=document.createElement("div");
            
            productimage.classList.add("product-img");
            productname.classList.add("product-name","text-center");
            productprice.classList.add("product-price","text-center");

            productname.textContent=product.title.substring(0,12)+"...";
            productprice.textContent=`&#8377; ${product.price}`;
            
            const imageinsideproductimage=document.createElement("img");
            imageinsideproductimage.src=product.image;
            

        //append divs
        productimage.appendChild(imageinsideproductimage);
        productitem.appendChild(productimage);
        productitem.appendChild(productname);
        productitem.appendChild(productprice);
        
        productlist.appendChild(productitem);
        });

    }
    populateproducts();
});