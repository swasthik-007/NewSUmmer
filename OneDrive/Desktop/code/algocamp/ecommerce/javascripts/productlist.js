document.addEventListener("DOMContentLoaded", async() =>{
     async function fetchproducts(){
        const response=await axios.get("https://fakestoreapi.com/products");
        console.log(response.data);
        return response.data;
    
    }
    async function fetchproductsbycategory(category){
        const response=await axios.get(`https://fakestoreapi.com/products/category/${category}`);
        console.log(response.data);
        return response.data;
    }
    async function fetchcategories(){
        //this function is marked async so this will also return a promise
    const response= await fetch("https://fakestoreapi.com/products/categories");
    const data= await response.json();
    return data;
    }

    const downloadedproducts= await fetchproducts();

    async function populateproducts(flag ,customproducts){
        let products =customproducts;
       const queryparams=new URLSearchParams(window.location.search);
       const queryparamsobject=Object.fromEntries(queryparams.entries());
        if(flag==false){
           if(queryparamsobject['category']){
            products= await fetchproductsbycategory(queryparamsobject['category']);
           }else{

               products=await fetchproducts();
           }
        }
        // const products =await fetchproducts();
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
            productprice.textContent=`$${product.price}`;
            
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
    async function populatecategories(){
        const categories=await fetchcategories();
        const categorylist=document.getElementById("categorylist");
        categories.forEach(category=>{
        const categorylink=document.createElement("a");
        categorylink.classList.add("d-flex", "text-decoration-none");
        categorylink.textContent=category;
        categorylink.href=`productlist.html?category=${category}`;
        
        categorylist.appendChild(categorylink);
    
    });
    }
    async function downloadandpopulate(){
       Promise.all([populateproducts(false),populatecategories()])
        .then(()=> {
            const loaderbackdrop=document.getElementById("loaderbackdrop");
            loaderbackdrop.style.display = 'none';

        });
        
    }
    downloadandpopulate();


    const filtersearch=document.getElementById("search");
    filtersearch.addEventListener("click",async ()=>{
        const productlist=document.getElementById("productlist");
        const minprice =Number(document.getElementById("minprice").value);
        const maxprice =Number(document.getElementById("maxprice").value);
        const products= downloadedproducts;
        filteredproducts=products.filter(product=> product.price >= minprice && product.price <=maxprice)
        productlist.innerHTML= "";
        populateproducts(true,filteredproducts);
    });
    const resetfilter =document.getElementById("clear");
    resetfilter.addEventListener("click",()=>{
        window.location.reload();
    })
});