document.addEventListener("DOMContentLoaded",()=>{

    function preparewrapperdivforcartitems(product, productquantitymapping) {
        const orderdetailsproduct = document.createElement("div");
        orderdetailsproduct.classList.add("order-details-product", "d-flex", "flex-row");
    
        const orderdetailsproductimg = document.createElement("div");
        orderdetailsproductimg.classList.add("order-details-product-img", "d-flex");
        const image = document.createElement("img");
        image.src = product.image;
        orderdetailsproductimg.appendChild(image);
    
        const orderdetailsproductdata = document.createElement("div");
        orderdetailsproductdata.classList.add("order-details-product-data", "d-flex", "flex-column");
        const name = document.createElement("div");
        const price = document.createElement("div");
        name.textContent = product.name;
        price.textContent = product.price;
    
        orderdetailsproductdata.appendChild(name);
        orderdetailsproductdata.appendChild(price);
    
        const orderdetailsproductactions = document.createElement("div");
        orderdetailsproductactions.classList.add("order-details-product-actions", "d-flex", "flex-column");
        const orderdetailsproductquantity = document.createElement("div");
        orderdetailsproductquantity.classList.add("order-details-product-quantity");
        const quantitylabel = document.createElement("div");
        quantitylabel.textContent = "Quantity";
        quantitylabel.classList.add("fw-bold");
        const formgroup = document.createElement("div");
        formgroup.classList.add("form-group");
        const select = document.createElement("select");
        select.classList.add("form-select");
        for (let i = 1; i <= 10; i++) {
            const option = document.createElement("option");
            option.textContent = i;
            option.value = i;
            if (i == productquantitymapping[product.productId]) {
                option.selected = "true";
            }
            select.appendChild(option);
        }
        formgroup.appendChild(select);
        orderdetailsproductquantity.appendChild(quantitylabel);
        orderdetailsproductquantity.appendChild(formgroup);
        orderdetailsproductactions.appendChild(orderdetailsproductquantity);
        const remove = document.createElement("button");
        remove.classList.add("order-details-product-remove", "btn", "btn-danger");
        remove.textContent = "Remove";
        orderdetailsproductactions.appendChild(remove);
    
        const hr = document.createElement("hr");
        orderdetailsproduct.appendChild(orderdetailsproductimg);
        orderdetailsproduct.appendChild(orderdetailsproductdata);
        orderdetailsproduct.appendChild(orderdetailsproductactions);
        document.getElementById("orderdetails").appendChild(orderdetailsproduct);
        document.getElementById("orderdetails").appendChild(hr);
    }
    


    async function populatecart(){
        const cart=await fetchcardsbyid(1);
        const cartproducts= cart.products;
        const productquantitymapping= {};
        console.log(cartproducts);
        const cartproductdownloadpromise=cartproducts.map(product =>{
            productquantitymapping[product.productId]=product.quantity;
            return fetchproductbyid(product.productId);
        });
        const products=await Promise.all(cartproductdownloadpromise);
        let totalprice=0;
        products.forEach(product =>{
            preparewrapperdivforcartitems(product,productquantitymapping);
            totalprice +=product.price * productquantitymapping[product.id];

        });
        document.getElementById("totalprice").textContent=totalprice;
        document.getElementById("netprice").textContent=totalprice-10;
   

        removeloader();
    }
    populatecart();
});