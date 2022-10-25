import { Col, Row } from "antd";

export default function ({
  A, B
}) {
  return (
    <Row gutter={[0, 0]}>
      <Col lg={24} xl={12}>
        {A}
      </Col>
      <Col lg={24} xl={12}>
        {B}
      </Col>
    </Row>
  )
}