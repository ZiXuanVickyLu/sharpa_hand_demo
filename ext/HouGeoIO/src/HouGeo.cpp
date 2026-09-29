#include <HouGeo.h>
#include <fstream>
#include <cassert>

using namespace hou;
using namespace Eigen;
template <typename... Ts>
struct Overload : Ts...
{
    using Ts::operator()...;
};

template <typename... Ts>
Overload(Ts...) -> Overload<Ts...>;

const void *HouGeo::Attribute::GetRawPointer() const
{
    const void *ptr;
    std::visit(
        Overload{
            [&](const MatrixXf &mat) { ptr = mat.data(); },
            [&](const MatrixXd &mat) { ptr = mat.data(); },
            [&](const MatrixXi &mat) { ptr = mat.data(); }},
        data_);

    return ptr;
}

void *HouGeo::Attribute::GetRawPointer()
{
    void *ptr;
    std::visit(
        Overload{
            [&](MatrixXf &mat) { ptr = mat.data(); },
            [&](MatrixXd &mat) { ptr = mat.data(); },
            [&](MatrixXi &mat) { ptr = mat.data(); }},
        data_);

    return ptr;
}

json::ObjectPtr toObject(json::ArrayPtr a)
{
    json::ObjectPtr o = json::Object::create();

    int numElements = (int)a->size();
    for (int i = 0; i < numElements; i += 2)
    {
        if (a->getValue(i).isString())
        {
            std::string key = a->get<std::string>(i);
            json::Value value = a->getValue(i + 1);
            o->append(key, value);
        }
    }

    return o;
}

void loadAttributeRawData(
    const int dstTupleSize,
    const int dstComponentSize,
    const int elementCount,
    char *data,
    json::ObjectPtr attrData)
{
    if (!attrData->hasKey("values"))
    {
        std::runtime_error("your attribute without value??? check your file");
        return;
    }

    json::ObjectPtr values = toObject(attrData->getArray("values"));
    if (values->hasKey("rawpagedata"))
    {
        int elementsPerPage = values->get<int>("pagesize");

        // one pack is a sequence of components
        // packing is used to describe in which sequence components are written to the file
        // packing allows to store vectors as list of structs or struct of lists.
        std::vector<ubyte> attrPacking;
        if (values->hasKey("packing"))
        {
            json::ArrayPtr packingArray = values->getArray("packing");
            int psize = (int)packingArray->size();
            for (int i = 0; i < psize; ++i)
            {
                attrPacking.push_back(packingArray->get<ubyte>(i));
            }
        }
        else
        {
            attrPacking.push_back(dstTupleSize);
        }
        // constantpageflags is an array which
        // contains an array for each pack
        // each of those per pack arrays contains flags for each page
        // which tell us wether the pack is constant for this page
        std::vector<std::vector<bool>> constantPageFlagsPerPack;

        // to make things even more fun, some packs can be constant
        // and this may be different per page - oh boy
        if (values->hasKey("constantpageflags"))
        {
            json::ArrayPtr constantPageFlags = values->getArray("constantpageflags");

            // for each pack
            int i = 0;
            for (auto it = attrPacking.begin(); it != attrPacking.end(); ++it, ++i)
            {
                constantPageFlagsPerPack.push_back(std::vector<bool>());

                // get array which tells us for each page if the pack is constant
                json::ArrayPtr packConstantFlags = constantPageFlags->getArray(i);

                for (int j = 0; j < packConstantFlags->size(); ++j)
                    constantPageFlagsPerPack.back().push_back(packConstantFlags->get<bool>(j));
            }
        }
        else
        {
            for (int j = 0; j < dstTupleSize; ++j) constantPageFlagsPerPack.push_back(std::vector<bool>());
        }

        json::ArrayPtr rawPageData = values->getArray("rawpagedata");

        // we need to repack - which when done in a generic way looks like a pain in the butt ======

        int elementsRemaining = elementCount;
        // qDebug() << "numElements " << ;
        // qDebug() << "rawPageData->size() " << (int)rawPageData->size();
        // qDebug() << "attrTupleSize " << attrTupleSize;

        // process each page
        int pageIndex = 0;
        int pageStartIndex = 0;
        while (elementsRemaining > 0)
        {
            int pageStartElement = pageIndex * elementsPerPage;
            size_t numElements = std::min(elementsRemaining, elementsPerPage);

            // process each pack
            int packIndex = 0;
            ubyte startComponentIndex = 0;
            for (std::vector<ubyte>::iterator it = attrPacking.begin(); it != attrPacking.end(); ++it, ++packIndex)
            {
                ubyte pack = *it;
                size_t maxPack = std::min((int)pack, std::max(0, dstTupleSize - startComponentIndex));

                if (maxPack == 0) break;

                // is pack for current page constant?
                bool isConstant = constantPageFlagsPerPack[packIndex].empty()
                                      ? false
                                      : constantPageFlagsPerPack[packIndex][pageIndex];
                // qDebug() << "constant? " << isConstant;

                // if pack is constant only the first element is given, this is the reference
                // find element index where the new page starts
                size_t elementIndex = pageStartIndex;

                // now iterate over all elements of current page and get values from current pack
                for (size_t i = 0; i < numElements; ++i)
                {
                    // we update elementIndex only if pack is varying within current page
                    // otherwise we will just keep pointing to the reference element
                    if (!isConstant)
                        // get page element index into rawpagedata for current pack
                        // we can do pageStartElement*attrTupleSize because packing doesnt matter for past pages
                        elementIndex = pageStartIndex + i * pack;
                    // qDebug() << "elementIndex " << elementIndex;
                    // qDebug() << "pageStartElement " << pageStartElement;
                    // qDebug() << "attrTupleSize " << attrTupleSize;
                    // qDebug() << "i " << i;
                    // qDebug() << "pack " << pack;

                    // get global element index for writing into our dense array
                    size_t destElementIndex = (pageStartElement + i) * dstTupleSize;

                    // for each component of current pack
                    for (size_t component = 0; component < maxPack; ++component)
                        // get component value from current rawpagedata
                        // and copy that component to the location of that component in dense array
                        // TODO: uniform arrays!
                        rawPageData->getValue(elementIndex + component)
                            .cpyTo((char *)&(
                                data[(destElementIndex + startComponentIndex + component) * dstComponentSize]));
                }

                startComponentIndex += pack;
                if (!isConstant)
                    pageStartIndex += numElements * pack;
                else
                    pageStartIndex += pack;
            }

            elementsRemaining -= numElements;
            pageStartElement += numElements;

            // proceed next page
            ++pageIndex;
        }
    }
}

HouGeo::Topology::Ptr HouIO::LoadTopology(json::ObjectPtr o)
{
    HouGeo::Topology::Ptr top = std::make_shared<HouGeo::Topology>();
    if (o->hasKey("pointref"))
    {
        json::ObjectPtr pointref = toObject(o->getArray("pointref"));
        if (pointref->hasKey("indices"))
        {
            json::ArrayPtr indices = pointref->getArray("indices");
            sint64 numElements = indices->size();
            top->indices_.resize(numElements);
            for (int i = 0; i < numElements; ++i)
            {
                top->indices_[i] = indices->get<int>(i);
            }
        }
    }
    return top;
}

void LoadPrimBy_r_v(
    json::ArrayPtr r_v_data,
    const int &n_p, // nPrimitive
    Eigen::VectorXi *nverticesPerPrim)
{
    Eigen::VectorXi &n_v = *nverticesPerPrim;
    n_v.resize(n_p);

    int n_r_v = r_v_data->size();

    int offset = 0;
    for (int i = 0; i < n_r_v / 2; ++i)
    {
        int nPrim = r_v_data->get<int>(2 * i);
        int nPrim_with_same_vertice = r_v_data->get<int>(2 * i + 1);
        for (int j = 0; j < nPrim_with_same_vertice; ++j)
        {
            n_v(offset) = nPrim;
            ++offset;
        }
    }
}

void LoadPrimBy_n_v(
    json::ArrayPtr n_v_data,
    const int &n_p, // nPrimitive
    Eigen::VectorXi *nverticesPerPrim)
{
    assert(n_p == n_v_data->size());

    Eigen::VectorXi &n_v = *nverticesPerPrim;
    n_v.resize(n_p);
    for (int i = 0; i < n_p; ++i)
    {
        n_v(i) = n_v_data->get<int>(i);
    }
}

HouGeo::Primitive::Ptr HouIO::LoadPolyPrimitive(hou::json::ArrayPtr primitive)
{
    HouGeo::Primitive::Ptr prim = std::make_shared<HouGeo::Primitive>();
    int size = primitive->size();

    json::ObjectPtr primdef = toObject(primitive->getArray(0));

    prim->type_ = primdef->get<std::string>("type");

    json::ObjectPtr primdata = toObject(primitive->getArray(1));

    prim->s_v_ = primdata->get<int>("s_v");
    prim->n_p_ = primdata->get<int>("n_p");

    if (primdata->hasKey("r_v"))
    {
        json::ArrayPtr r_v_data = primdata->getArray("r_v");
        LoadPrimBy_r_v(r_v_data, prim->n_p_, &prim->nverticesPerPrim_);
    }
    else if (primdata->hasKey("n_v"))
    {
        json::ArrayPtr r_v_data = primdata->getArray("n_v");
        LoadPrimBy_n_v(r_v_data, prim->n_p_, &prim->nverticesPerPrim_);
    }

    return prim;
}

HouGeo::Attribute::Ptr HouIO::LoadAttribute(hou::json::ArrayPtr attribute, const int32_t elementCount)
{
    json::ObjectPtr attrDef = toObject(attribute->getArray(0));
    json::ObjectPtr attrData = toObject(attribute->getArray(1));

    HouGeo::Attribute::Ptr attr = std::make_shared<HouGeo::Attribute>();

    attr->name_ = attrDef->get<std::string>("name");
    attr->type_ = attrDef->get<std::string>("type");
    attr->num_element_ = elementCount;

    if (attr->type_ == "numeric")
    {
        attr->storage_ = attrData->get<std::string>("storage");
        attr->tuple_size_ = attrData->get<int>("size");

        int dstTupleSize = attr->tuple_size_;
        int dstComponentSize = 0;
        {
            if (attr->storage_ == "fpreal32")
            {
                dstComponentSize = sizeof(float);

                Eigen::MatrixXf mat;
                mat.resize(dstTupleSize, elementCount);
                attr->data_ = mat;
            }
            else if (attr->storage_ == "fpreal64")
            {
                dstComponentSize = sizeof(double);

                Eigen::MatrixXd mat;
                mat.resize(dstTupleSize, elementCount);
                attr->data_ = mat;
            }
            else if (attr->storage_ == "int32")
            {
                dstComponentSize = sizeof(int);

                Eigen::MatrixXi mat;
                mat.resize(dstTupleSize, elementCount);
                attr->data_ = mat;
            }
        }

        char *data_ptr = (char *)attr->GetRawPointer();

        loadAttributeRawData(dstTupleSize, dstComponentSize, elementCount, data_ptr, attrData);
    }
    else if (attr->type_ == "string")
    {
        // forget it;
        throw std::runtime_error("Find string type attribute!!!!");
    }
    else if (attr->type_ == "arraydata")
    {
        // only support one arraydata for now
        // size is 1 x n tuple
        attr->storage_ = attrData->get<std::string>("storage");
        int num_arrays = attrData->get<int>("size");

        // we only support one arraydata for now
        assert(num_arrays == 1);

        json::ArrayPtr array = attrData->getArray("values")->getArray(0);

        // special case for arraydata, tuple_size_ = 1, num_element_ = array->size();
        attr->tuple_size_ = 1;
        attr->num_element_ = array->size();

        int dstTupleSize = array->size();

        {
            if (attr->storage_ == "fpreal32")
            {
                Eigen::MatrixXf mat;
                mat.resize(dstTupleSize, 1);

                for (int i = 0; i < dstTupleSize; ++i)
                {
                    mat(i, 0) = array->get<float>(i);
                }

                attr->data_ = mat;
            }
            else if (attr->storage_ == "fpreal64")
            {
                std::cout << "we don't support double type arraydata attribute now" << std::endl;

                throw;
            }
            else if (attr->storage_ == "int32")
            {
                Eigen::MatrixXi mat;
                mat.resize(dstTupleSize, 1);

                for (int i = 0; i < dstTupleSize; ++i)
                {
                    mat(i, 0) = array->get<int>(i);
                }

                attr->data_ = mat;
            }
        }
    }

    return attr;
}

void HouIO::ExportAttribute(json::BinaryWriter *g_writer, HouGeo::Attribute::CPtr attr)
{
    if (!attr)
    {
        return;
    }

    const std::string &type = attr->type_;
    const std::string &name = attr->name_;
    const std::string &storage = attr->storage_;
    const int tuple_size = attr->tuple_size_;
    const int num_elements = attr->num_element_;

    g_writer->jsonBeginArray();
    {
        // attribute definition
        g_writer->jsonBeginArray();
        {
            g_writer->jsonString("scope");
            g_writer->jsonString("public");

            g_writer->jsonString("type");
            g_writer->jsonString(type);

            g_writer->jsonString("name");
            g_writer->jsonString(name);

            g_writer->jsonString("options");
            g_writer->jsonBeginMap();
            {
                if (name == "P")
                {
                    g_writer->jsonKey("type");
                    g_writer->jsonBeginMap();
                    {
                        g_writer->jsonKey("type");
                        g_writer->jsonString("string");
                        g_writer->jsonKey("value");
                        g_writer->jsonString("point");
                    }
                    g_writer->jsonEndMap();
                }
                else
                {
                    // no else;
                }
            }
            g_writer->jsonEndMap();
        }
        g_writer->jsonEndArray(); // definition

        // attribute data
        g_writer->jsonBeginArray();
        {
            if (type == "numeric")
            {
                g_writer->jsonString("size");
                g_writer->jsonInt(tuple_size);

                g_writer->jsonString("storage");
                g_writer->jsonString(storage);

                g_writer->jsonString("values");
                g_writer->jsonBeginArray();
                {
                    g_writer->jsonString("size");
                    g_writer->jsonInt(tuple_size);

                    g_writer->jsonString("storage");
                    g_writer->jsonString(storage);

                    g_writer->jsonString("pagesize");
                    g_writer->jsonInt(1024);

                    g_writer->jsonString("rawpagedata");
                    if (storage == "fpreal32")
                    {
                        g_writer->jsonUniformArray<real32>(
                            (const real32 *)attr->GetRawPointer(), num_elements * tuple_size);
                    }
                    else if (storage == "fpreal64")
                    {
                        g_writer->jsonUniformArray<real64>(
                            (const real64 *)attr->GetRawPointer(), num_elements * tuple_size);
                    }
                    else if (storage == "int32")
                    {
                        g_writer->jsonUniformArray<sint32>(
                            (const sint32 *)attr->GetRawPointer(), num_elements * tuple_size);
                    }
                }
                g_writer->jsonEndArray();
            }
            else if (type == "string")
            {
                // forget it;
                throw std::runtime_error("Find string type attribute!!!!");
            }
            else if (type == "arraydata")
            {
                g_writer->jsonString("size");
                g_writer->jsonInt(tuple_size);

                g_writer->jsonString("storage");
                g_writer->jsonString(storage);

                g_writer->jsonString("values");

                g_writer->jsonBeginArray();
                {
                    if (storage == "fpreal32")
                    {
                        g_writer->jsonUniformArray<real32>(
                            (const real32 *)attr->GetRawPointer(), num_elements * tuple_size);
                    }
                    else if (storage == "fpreal64")
                    {
                        std::cout << "we don't support double type arraydata attribute now" << std::endl;
                        throw;
                    }
                    else if (storage == "int32")
                    {
                        g_writer->jsonUniformArray<sint32>(
                            (const sint32 *)attr->GetRawPointer(), num_elements * tuple_size);
                    }
                }
                g_writer->jsonEndArray();
            }
        }
        g_writer->jsonEndArray(); // data
    }
    g_writer->jsonEndArray();
}

bool HouIO::ImportHouGeo(const std::string &file, HouGeo *geo)
{
    std::ifstream in(file.c_str(), std::ios_base::in | std::ios_base::binary);

    json::JSONReader reader;
    json::Parser p;

    if (!p.parse(&in, &reader))
    {
        return false;
    }
    json::ObjectPtr o = toObject(reader.getRoot().asArray());
    int numVertices = 0;
    int numPoints = 0;
    int numPrimitives = 0;
    if (o->hasKey("pointcount"))
    {
        numPoints = o->get<int>("pointcount", 0);
    }
    if (o->hasKey("vertexcount"))
    {
        numVertices = o->get<int>("vertexcount", 0);
    }
    if (o->hasKey("primitivecount"))
    {
        numPrimitives = o->get<int>("primitivecount", 0);
    }

    if (o->hasKey("attributes"))
    {
        json::ObjectPtr attributes = toObject(o->getArray("attributes"));
        if (attributes->hasKey("pointattributes"))
        {
            json::ArrayPtr pointAttributes = attributes->getArray("pointattributes");
            sint64 numPointAttributes = pointAttributes->size();
            for (int i = 0; i < numPointAttributes; ++i)
            {
                json::ArrayPtr pointAttribute = pointAttributes->getArray(i);
                HouGeo::Attribute::Ptr attr = LoadAttribute(pointAttribute, numPoints);
                geo->point_attributes_.insert(std::make_pair(attr->GetName(), attr));
            }
        }
        if (attributes->hasKey("vertexattributes"))
        {
            json::ArrayPtr vertexAttributes = attributes->getArray("vertexattributes");
            sint64 numVertexAttributes = vertexAttributes->size();
            for (int i = 0; i < numVertexAttributes; ++i)
            {
                json::ArrayPtr vertexAttribute = vertexAttributes->getArray(i);
                HouGeo::Attribute::Ptr attr = LoadAttribute(vertexAttribute, numVertices);
                geo->vertex_attributes_.insert(std::make_pair(attr->GetName(), attr));
            }
        }
        if (attributes->hasKey("primitiveattributes"))
        {
            json::ArrayPtr primitiveAttributes = attributes->getArray("primitiveattributes");
            sint64 numPrimitiveAttributes = primitiveAttributes->size();
            for (int i = 0; i < numPrimitiveAttributes; ++i)
            {
                json::ArrayPtr primitiveAttribute = primitiveAttributes->getArray(i);
                HouGeo::Attribute::Ptr attr = LoadAttribute(primitiveAttribute, numPrimitives);
                geo->primitive_attributes_.insert(std::make_pair(attr->GetName(), attr));
            }
        }
        if (attributes->hasKey("globalattributes"))
        {
            json::ArrayPtr globalAttributes = attributes->getArray("globalattributes");
            sint64 numGlobalAttributes = globalAttributes->size();
            for (int i = 0; i < numGlobalAttributes; ++i)
            {
                json::ArrayPtr globalAttribute = globalAttributes->getArray(i);
                // TODO: element count argument, how many?!
                HouGeo::Attribute::Ptr attr = LoadAttribute(globalAttribute, 1);
                geo->global_attributes_.insert(std::make_pair(attr->GetName(), attr));
            }
        }
    }

    if (o->hasKey("topology"))
    {
        HouGeo::Topology::Ptr top = LoadTopology(toObject(o->getArray("topology")));
        geo->topology_ = top;
    }

    if (o->hasKey("primitives"))
    {
        json::ArrayPtr primitives = o->getArray("primitives");
        int numPrimArray = primitives->size();
        for (int j = 0; j < numPrimArray; ++j)
        {
            json::ArrayPtr primitive = primitives->getArray(j);

            HouGeo::Primitive::Ptr prim = LoadPolyPrimitive(primitive);
            geo->primitive_ = prim;
        }
    }
    return true;
}

bool HouIO::ExportHouGeo(const std::string &file, const HouGeo &geo)
{
    std::ofstream out(file.c_str(), std::ios_base::out | std::ios_base::binary);

    if (out.fail())
    {
        std::cerr << "Unable to open .bgeo file: " << file << " !!!!" << std::endl;
        throw;
    }

    json::BinaryWriter *g_writer;

    g_writer = new json::BinaryWriter(&out);
    int PointCount = 0;
    int VertexCount = 0;
    int PrimitiveCount = 0;
    {
        if (geo.point_attributes_.find("P") != geo.point_attributes_.end())
        {
            PointCount = geo.point_attributes_.at("P")->GetNumElment();
        }
        else
        {
            PointCount = 0;
        }

        if (geo.topology_ != nullptr)
        {
            VertexCount = geo.topology_->indices_.size();
        }

        if (geo.primitive_ != nullptr)
        {
            PrimitiveCount = geo.primitive_->GetNumPrimitive();
        }
    }

    g_writer->jsonBeginArray();
    {
        g_writer->jsonString("pointcount");
        g_writer->jsonInt(PointCount);

        g_writer->jsonString("vertexcount");
        g_writer->jsonInt(VertexCount);

        g_writer->jsonString("primitivecount");
        g_writer->jsonInt(PrimitiveCount);

        g_writer->jsonString("topology");
        g_writer->jsonBeginArray();
        {
            g_writer->jsonString("pointref");
            g_writer->jsonBeginArray();
            {
                g_writer->jsonString("indices");
                g_writer->jsonUniformArray<int>(geo.topology_->indices_.data(), VertexCount);
            }
            g_writer->jsonEndArray();
        }
        g_writer->jsonEndArray();

        g_writer->jsonString("attributes");
        g_writer->jsonBeginArray();
        {
            if (geo.point_attributes_.size() != 0)
            {
                g_writer->jsonString("pointattributes");
                g_writer->jsonBeginArray();
                {
                    for (const auto &pair : geo.point_attributes_)
                    {
                        ExportAttribute(g_writer, pair.second);
                    }
                }
                g_writer->jsonEndArray();
            }

            if (geo.primitive_attributes_.size() != 0)
            {
                g_writer->jsonString("primitiveattributes");
                g_writer->jsonBeginArray();
                {
                    for (const auto &pair : geo.primitive_attributes_)
                    {
                        ExportAttribute(g_writer, pair.second);
                    }
                }
                g_writer->jsonEndArray();
            }

            if (geo.vertex_attributes_.size() != 0)
            {
                g_writer->jsonString("vertexattributes");
                g_writer->jsonBeginArray();
                {
                    for (const auto &pair : geo.vertex_attributes_)
                    {
                        ExportAttribute(g_writer, pair.second);
                    }
                }
                g_writer->jsonEndArray();
            }

            if (geo.global_attributes_.size() != 0)
            {
                g_writer->jsonString("globalattributes");
                g_writer->jsonBeginArray();
                {
                    for (const auto &pair : geo.global_attributes_)
                    {
                        ExportAttribute(g_writer, pair.second);
                    }
                }
                g_writer->jsonEndArray();
            }
        }
        g_writer->jsonEndArray();

        g_writer->jsonString("primitives");
        g_writer->jsonBeginArray();
        {
            if (PrimitiveCount > 0)
            {
                g_writer->jsonBeginArray();
                {
                    // def
                    g_writer->jsonBeginArray();
                    {
                        g_writer->jsonString("type");
                        g_writer->jsonString(geo.primitive_->type_);
                    }
                    g_writer->jsonEndArray();

                    g_writer->jsonBeginArray();
                    {
                        g_writer->jsonString("s_v");
                        g_writer->jsonInt32(0);
                        g_writer->jsonString("n_p");
                        g_writer->jsonInt32(PrimitiveCount);

                        if (geo.primitive_->type_ == "h_r")
                        {
                        }
                        else if (geo.primitive_->type_ == "t_r")
                        {
                        }
                        else
                        {
                            g_writer->jsonString("n_v");
                            g_writer->jsonUniformArray<int>(
                                (const int *)geo.primitive_->GetNumVerticesArrayRawPointer(), PrimitiveCount);
                        }
                    }
                    g_writer->jsonEndArray();
                }
                g_writer->jsonEndArray();
            }
        }
        g_writer->jsonEndArray();
    }
    g_writer->jsonEndArray();

    delete g_writer;
    return true;
}
